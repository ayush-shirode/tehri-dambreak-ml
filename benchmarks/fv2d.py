"""Experiment C: 2D depth-averaged shallow-water finite-volume solver (HLL flux,
hydrostatic reconstruction of Audusse et al. 2004 for wet/dry and topography,
semi-implicit Manning friction) applied to a synthetic sinuous valley with an
instantaneous full-width dam breach. Used to generate the scenario ensemble."""
import numpy as np
from numba import njit
g = 9.81
HMIN = 1e-3

def make_valley(S0, nx=300, ny=60, dx=40.0):
    x = (np.arange(nx)+0.5)*dx; y = (np.arange(ny)+0.5)*dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    yc = 0.5*ny*dx + 300*np.sin(2*np.pi*X/4000.0)       # meandering thalweg
    d = np.abs(Y-yc)
    z = S0*(nx*dx - X) + 0.02*d + 1.5e-5*d**2           # longitudinal slope + valley sides
    return X, Y, z

@njit(cache=True)
def hll(hl, ul, vl, hr, ur, vr):
    # x-normal HLL flux for (h, hu, hv)
    cl = np.sqrt(g*hl); cr = np.sqrt(g*hr)
    if hl < HMIN and hr < HMIN:
        return 0.0, 0.0, 0.0
    if hl < HMIN:
        sl = ur-2*cr; sr = ur+cr
    elif hr < HMIN:
        sl = ul-cl; sr = ul+2*cl
    else:
        sl = min(ul-cl, ur-cr); sr = max(ul+cl, ur+cr)
    fl0 = hl*ul; fl1 = hl*ul*ul+0.5*g*hl*hl; fl2 = hl*ul*vl
    fr0 = hr*ur; fr1 = hr*ur*ur+0.5*g*hr*hr; fr2 = hr*ur*vr
    if sl >= 0: return fl0, fl1, fl2
    if sr <= 0: return fr0, fr1, fr2
    den = sr-sl
    return ((sr*fl0-sl*fr0+sl*sr*(hr-hl))/den,
            (sr*fl1-sl*fr1+sl*sr*(hr*ur-hl*ul))/den,
            (sr*fl2-sl*fr2+sl*sr*(hr*vr-hl*vl))/den)

@njit(cache=True)
def step(h, hu, hv, z, dx, dt, n):
    nx, ny = h.shape
    dh = np.zeros_like(h); dqu = np.zeros_like(h); dqv = np.zeros_like(h)
    # x-faces
    for i in range(nx-1):
        for j in range(ny):
            hL, hR = h[i, j], h[i+1, j]
            uL = hu[i, j]/hL if hL > HMIN else 0.0; vL = hv[i, j]/hL if hL > HMIN else 0.0
            uR = hu[i+1, j]/hR if hR > HMIN else 0.0; vR = hv[i+1, j]/hR if hR > HMIN else 0.0
            zf = max(z[i, j], z[i+1, j])
            hl = max(0.0, hL+z[i, j]-zf); hr = max(0.0, hR+z[i+1, j]-zf)
            f0, f1, f2 = hll(hl, uL, vL, hr, uR, vR)
            # hydrostatic reconstruction source correction
            sL = 0.5*g*(hL*hL-hl*hl); sR = 0.5*g*(hR*hR-hr*hr)
            dh[i, j] -= f0/dx; dqu[i, j] -= (f1+sL)/dx; dqv[i, j] -= f2/dx
            dh[i+1, j] += f0/dx; dqu[i+1, j] += (f1+sR)/dx; dqv[i+1, j] += f2/dx
    # y-faces (rotate: normal velocity = v)
    for i in range(nx):
        for j in range(ny-1):
            hL, hR = h[i, j], h[i, j+1]
            uL = hu[i, j]/hL if hL > HMIN else 0.0; vL = hv[i, j]/hL if hL > HMIN else 0.0
            uR = hu[i, j+1]/hR if hR > HMIN else 0.0; vR = hv[i, j+1]/hR if hR > HMIN else 0.0
            zf = max(z[i, j], z[i, j+1])
            hl = max(0.0, hL+z[i, j]-zf); hr = max(0.0, hR+z[i, j+1]-zf)
            f0, f2, f1 = hll(hl, vL, uL, hr, vR, uR)
            sL = 0.5*g*(hL*hL-hl*hl); sR = 0.5*g*(hR*hR-hr*hr)
            dh[i, j] -= f0/dx; dqv[i, j] -= (f2+sL)/dx; dqu[i, j] -= f1/dx
            dh[i, j+1] += f0/dx; dqv[i, j+1] += (f2+sR)/dx; dqu[i, j+1] += f1/dx
    # closed walls (upstream end, valley sides): reflective hydrostatic pressure
    for j in range(ny):
        dqu[0, j] += 0.5*g*h[0, j]**2/dx
        # transmissive outflow at downstream face: physical flux of last cell
        hl = h[nx-1, j]
        if hl > HMIN:
            u = max(hu[nx-1, j]/hl, 0.0); v = hv[nx-1, j]/hl
            dh[nx-1, j] -= hl*u/dx; dqu[nx-1, j] -= (hl*u*u+0.5*g*hl*hl)/dx; dqv[nx-1, j] -= hl*u*v/dx
    for i in range(nx):
        dqv[i, 0] += 0.5*g*h[i, 0]**2/dx
        dqv[i, ny-1] -= 0.5*g*h[i, ny-1]**2/dx
    smax = 0.0
    for i in range(nx):
        for j in range(ny):
            hn = h[i, j]+dt*dh[i, j]
            qu = hu[i, j]+dt*dqu[i, j]; qv = hv[i, j]+dt*dqv[i, j]
            if hn < HMIN:
                h[i, j] = max(hn, 0.0); hu[i, j] = 0.0; hv[i, j] = 0.0; continue
            # semi-implicit Manning friction
            sp = np.sqrt(qu*qu+qv*qv)/hn
            fac = 1.0 + dt*g*n*n*sp/hn**(4.0/3.0)
            h[i, j] = hn; hu[i, j] = qu/fac; hv[i, j] = qv/fac
            c = np.sqrt(qu*qu+qv*qv)/fac/hn + np.sqrt(g*hn)
            if c > smax: smax = c
    return smax

def run(Hd, n, S0, Tend=3600.0, xdam=2000.0, cfl=0.45, gauges=(4000, 6000, 8000, 10000), harr=0.3):
    X, Y, z = make_valley(S0); dx = X[1, 0]-X[0, 0]
    zdam = z[np.argmin(abs(X[:, 0]-xdam)), :].min()     # thalweg bed level at dam
    wl = zdam + Hd
    h = np.where(X < xdam, np.maximum(wl-z, 0.0), 0.0); hu = np.zeros_like(h); hv = np.zeros_like(h)
    V0 = h.sum()*dx*dx
    hmax = h.copy(); arr = np.full(h.shape, np.nan)
    t = 0.0; smax = np.sqrt(g*Hd)
    gi = [int(gx//dx) for gx in gauges]
    while t < Tend:
        dt = min(cfl*dx/max(smax, 1e-3), Tend-t)
        smax = step(h, hu, hv, z, dx, dt, n)
        t += dt
        np.maximum(hmax, h, out=hmax)
        newly = np.isnan(arr) & (h > harr) & (X >= xdam); arr[newly] = t
    return dict(X=X, Y=Y, z=z, hmax=hmax, arr=arr, V0=V0, gi=gi, xdam=xdam)

if __name__ == "__main__":
    import time
    t0 = time.time(); r = run(30.0, 0.04, 0.002)
    print("runtime", time.time()-t0, "V0 (Mm3)", r["V0"]/1e6)
    for gi in r["gi"]:
        col = r["hmax"][gi]; print(gi, col.max(), np.nanmin(r["arr"][gi]))
    print("inund area km2", ((r["hmax"] > 0.3) & (r["X"] >= 2000)).sum()*40*40/1e6)
