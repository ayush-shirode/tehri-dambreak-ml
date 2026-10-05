"""2D shallow-water FV solver on a DEM (HLL + hydrostatic reconstruction, semi-implicit
Manning friction, open boundaries), driven by a breach hydrograph injected as a mass
source over the breach cells. Parallelised with numba prange (face-flux arrays, no races)."""
import numpy as np
from numba import njit, prange
g = 9.81
HMIN = 1e-3

@njit(inline="always")
def hll(hl, ul, vl, hr, ur, vr):
    if hl < HMIN and hr < HMIN: return 0.0, 0.0, 0.0
    cl = np.sqrt(g*hl); cr = np.sqrt(g*hr)
    if hl < HMIN: sl = ur-2*cr; sr = ur+cr
    elif hr < HMIN: sl = ul-cl; sr = ul+2*cl
    else: sl = min(ul-cl, ur-cr); sr = max(ul+cl, ur+cr)
    fl0 = hl*ul; fl1 = hl*ul*ul+0.5*g*hl*hl; fl2 = hl*ul*vl
    fr0 = hr*ur; fr1 = hr*ur*ur+0.5*g*hr*hr; fr2 = hr*ur*vr
    if sl >= 0: return fl0, fl1, fl2
    if sr <= 0: return fr0, fr1, fr2
    d = sr-sl
    return ((sr*fl0-sl*fr0+sl*sr*(hr-hl))/d, (sr*fl1-sl*fr1+sl*sr*(hr*ur-hl*ul))/d, (sr*fl2-sl*fr2+sl*sr*(hr*vr-hl*vl))/d)

@njit(inline="always")
def face(hL, quL, qvL, zL, hR, quR, qvR, zR, side):
    uL = quL/hL if hL > HMIN else 0.0; vL = qvL/hL if hL > HMIN else 0.0
    uR = quR/hR if hR > HMIN else 0.0; vR = qvR/hR if hR > HMIN else 0.0
    if side == -1: uL = min(uL, 0.0)      # open boundary on the low side: outflow only
    if side == 1: uR = max(uR, 0.0)       # open boundary on the high side
    zf = max(zL, zR)
    hl = max(0.0, hL+zL-zf); hr = max(0.0, hR+zR-zf)
    f0, f1, f2 = hll(hl, uL, vL, hr, uR, vR)
    return f0, f1, f2, 0.5*g*(hL*hL-hl*hl), 0.5*g*(hR*hR-hr*hr)

@njit(parallel=True)
def step(h, hu, hv, z, dx, dt, n, src, qsrc):
    nx, ny = h.shape
    FX = np.zeros((3, nx+1, ny)); SXL = np.zeros((nx+1, ny)); SXR = np.zeros((nx+1, ny))
    for i in range(nx+1):
        ia = i-1; ib = i; side = 0
        if i == 0:
            ia = 0; side = -1
        if i == nx:
            ib = nx-1; side = 1
        for j in range(ny):
            f0, f1, f2, sl, sr = face(h[ia, j], hu[ia, j], hv[ia, j], z[ia, j], h[ib, j], hu[ib, j], hv[ib, j], z[ib, j], side)
            FX[0, i, j] = f0; FX[1, i, j] = f1; FX[2, i, j] = f2; SXL[i, j] = sl; SXR[i, j] = sr
    FY = np.zeros((3, nx, ny+1)); SYL = np.zeros((nx, ny+1)); SYR = np.zeros((nx, ny+1))
    for i in range(nx):
        for j in range(ny+1):
            ja = j-1; jb = j; side = 0
            if j == 0:
                ja = 0; side = -1
            if j == ny:
                jb = ny-1; side = 1
            f0, f2, f1, sl, sr = face(h[i, ja], hv[i, ja], hu[i, ja], z[i, ja], h[i, jb], hv[i, jb], hu[i, jb], z[i, jb], side)
            FY[0, i, j] = f0; FY[1, i, j] = f1; FY[2, i, j] = f2; SYL[i, j] = sl; SYR[i, j] = sr
    smax = np.zeros(nx)
    for i in prange(nx):
        for j in range(ny):
            dh = -(FX[0, i+1, j]-FX[0, i, j] + FY[0, i, j+1]-FY[0, i, j])/dx
            dqu = -((FX[1, i+1, j]+SXL[i+1, j]) - (FX[1, i, j]+SXR[i, j]) + FY[1, i, j+1]-FY[1, i, j])/dx
            dqv = -(FX[2, i+1, j]-FX[2, i, j] + (FY[2, i, j+1]+SYL[i, j+1]) - (FY[2, i, j]+SYR[i, j]))/dx
            if src[i, j]: dh += qsrc
            hn = h[i, j]+dt*dh; qu = hu[i, j]+dt*dqu; qv = hv[i, j]+dt*dqv
            if hn < HMIN:
                h[i, j] = max(hn, 0.0); hu[i, j] = 0.0; hv[i, j] = 0.0; continue
            sp = np.sqrt(qu*qu+qv*qv)/hn
            fac = 1.0 + dt*g*n*n*sp/hn**(4.0/3.0)
            h[i, j] = hn; hu[i, j] = qu/fac; hv[i, j] = qv/fac
            c = np.sqrt(qu*qu+qv*qv)/fac/hn + np.sqrt(g*hn)
            if c > smax[i]: smax[i] = c
    return smax.max()

def run(z, dx, src_mask, t_hyd, q_hyd, n, T, gauges, harr=1.0, cfl=0.45, h0=None):
    """z: bed (m); src_mask: breach-inflow cells; (t_hyd, q_hyd): hydrograph (m3/s).
    gauges: list of (i, j-array) cross-sections. Returns hmax, arrival-time field, gauge stats."""
    h = np.zeros_like(z) if h0 is None else h0.copy()
    hu = np.zeros_like(z); hv = np.zeros_like(z)
    hmax = h.copy(); arr = np.full(z.shape, np.nan)
    nsrc = src_mask.sum(); t = 0.0; smax = 10.0
    gpk = np.zeros(len(gauges)); garr = np.full(len(gauges), np.nan)
    vin = 0.0
    while t < T:
        dt = min(cfl*dx/max(smax, 1.0), T-t, 10.0)
        Q = np.interp(t+0.5*dt, t_hyd, q_hyd)
        vin += Q*dt
        smax = step(h, hu, hv, z, dx, dt, n, src_mask, Q/(nsrc*dx*dx))
        t += dt
        np.maximum(hmax, h, out=hmax)
        newly = np.isnan(arr) & (h > harr); arr[newly] = t
        for k, (gi, gj) in enumerate(gauges):
            gpk[k] = max(gpk[k], h[gi, gj].max())
            if np.isnan(garr[k]) and (h[gi, gj] > harr).any(): garr[k] = t
    return dict(hmax=hmax, arr=arr, gauge_peak=gpk, gauge_arr=garr, vin=vin, vfinal=h.sum()*dx*dx)
