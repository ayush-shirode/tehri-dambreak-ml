"""Experiment B: 2D weakly-compressible SPH (WCSPH) collapse of a water column on a dry
horizontal bed, compared with depth-averaged FV-SWE and Ritter's front speed.
Wendland C2 kernel, Tait EOS, Monaghan artificial viscosity, delta-SPH density diffusion,
dynamic boundary particles (fixed wall particles with evolving density)."""
import numpy as np, json, time, sys
from numba import njit
g = 9.81

H, A, LT = 1.0, 0.5, 6.0           # column height, column width, tank length (m)
dp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
hk = 1.3*dp; rho0 = 1000.0; c0 = 10*np.sqrt(g*H); gam = 7.0; B = rho0*c0**2/gam
alpha, delta = 0.02, 0.1
alphaD = 7.0/(4*np.pi*hk**2)

# fluid particles
xs = np.arange(dp/2, A, dp); ys = np.arange(dp/2, H, dp)
X, Y = np.meshgrid(xs, ys); fx = X.ravel(); fy = Y.ravel()
nf = fx.size
# wall particles: bottom and left wall, 3 layers
wx, wy = [], []
for k in range(3):
    xb = np.arange(-3*dp+dp/2, LT+dp/2, dp)
    wx += list(xb); wy += [-(k+0.5)*dp]*xb.size
    yb = np.arange(dp/2, 1.5*H, dp)
    wx += [-(k+0.5)*dp]*yb.size; wy += list(yb)
wx = np.array(wx); wy = np.array(wy)
px = np.concatenate([fx, wx]); py = np.concatenate([fy, wy])
N = px.size
isf = np.zeros(N, np.bool_); isf[:nf] = True
vx = np.zeros(N); vy = np.zeros(N)
# hydrostatic initial density
rho = np.full(N, rho0)
rho[:nf] = rho0*(1 + rho0*g*(H-fy)/B)**(1/gam)
m = rho0*dp*dp

@njit
def kgrad(r):
    q = r/hk
    if q >= 2.0 or r < 1e-12: return 0.0
    return alphaD*(-5.0*q*(1-0.5*q)**3)/hk/r   # dW/dr / r

@njit
def forces(px, py, vx, vy, rho, isf, cell, start, cnt, ncx, ncy, x0, y0):
    N = px.size
    ax = np.zeros(N); ay = np.zeros(N); drho = np.zeros(N)
    P = B*((rho/rho0)**gam - 1.0)
    cs = 2*hk
    for i in range(N):
        ci = int((px[i]-x0)/cs); cj = int((py[i]-y0)/cs)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                a = ci+di; b = cj+dj
                if a < 0 or b < 0 or a >= ncx or b >= ncy: continue
                c = a*ncy+b
                for kk in range(start[c], start[c]+cnt[c]):
                    j = cell[kk]
                    if j == i: continue
                    dx = px[i]-px[j]; dy = py[i]-py[j]
                    r = np.sqrt(dx*dx+dy*dy)
                    if r >= 2*hk: continue
                    F = kgrad(r)
                    dvx = vx[i]-vx[j]; dvy = vy[i]-vy[j]
                    vr = dvx*dx+dvy*dy
                    # continuity + delta-SPH diffusion (Molteni & Colagrossi 2009)
                    drho[i] += m*vr*F
                    if isf[i] and isf[j]:
                        drho[i] -= delta*hk*c0*2*(rho[j]-rho[i])*r*r*F/(r*r+0.01*hk*hk)*m/rho[j]
                    if not isf[i]: continue
                    # momentum
                    pij = P[i]/rho[i]**2 + P[j]/rho[j]**2
                    visc = 0.0
                    if vr < 0:
                        mu = hk*vr/(r*r+0.01*hk*hk)
                        visc = -alpha*c0*mu/(0.5*(rho[i]+rho[j]))
                    ax[i] += -m*(pij+visc)*dx*F
                    ay[i] += -m*(pij+visc)*dy*F
    for i in range(N):
        if isf[i]: ay[i] -= g
    return ax, ay, drho

def neighbour_grid(px, py):
    cs = 2*hk; x0 = px.min()-cs; y0 = py.min()-cs
    ncx = int((px.max()-x0)/cs)+2; ncy = int((py.max()-y0)/cs)+2
    cid = ((px-x0)//cs).astype(np.int64)*ncy + ((py-y0)//cs).astype(np.int64)
    order = np.argsort(cid, kind="stable")
    cnt = np.bincount(cid, minlength=ncx*ncy)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return order, start, cnt, ncx, ncy, x0, y0

tout = np.sqrt(H/g)*np.array([0.5, 1.0, 1.5, 2.0, 2.5])
Tend = tout[-1]; t = 0.0; front = []; snaps = {}
dt = 0.1*hk/(c0*1.1)
t0 = time.time(); step = 0; k = 0
while t < Tend:
    # symplectic (semi-implicit) Euler time stepping
    o, s, c, ncx, ncy, x0, y0 = neighbour_grid(px, py)
    ax, ay, dr = forces(px, py, vx, vy, rho, isf, o, s, c, ncx, ncy, x0, y0)
    vx = (vx + dt*ax)*isf; vy = (vy + dt*ay)*isf; rho = rho + dt*dr
    px = px + dt*vx; py = py + dt*vy
    # wall density floor to avoid spurious tension at walls
    rho[~isf] = np.maximum(rho[~isf], rho0)
    t += dt; step += 1
    if step % 50 == 0:
        sel = isf & (py < 0.1*H)
        front.append([t*np.sqrt(g/H), (px[sel].max())/A])
    if k < len(tout) and t >= tout[k]:
        snaps[f"{t*np.sqrt(g/H):.1f}"] = {"x": px[:nf].tolist(), "y": py[:nf].tolist(),
                                         "v": np.hypot(vx[:nf], vy[:nf]).tolist()}
        k += 1
print("steps", step, "time", time.time()-t0, "nf", nf, "N", N)
json.dump({"dp": dp, "front": front, "snaps": snaps, "nf": int(nf), "c0": c0},
          open(f"sph_{dp}.json", "w"))
