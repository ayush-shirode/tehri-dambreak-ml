"""Experiment A: 1D finite-volume shallow-water solver (HLL, MUSCL-minmod, SSP-RK2)
verified against Ritter (dry bed) and Stoker (wet bed) analytical dam-break solutions."""
import numpy as np, json
from scipy.optimize import brentq
g = 9.81
HDRY = 1e-6

def hll_flux(hL, uL, hR, uR):
    cL, cR = np.sqrt(g*hL), np.sqrt(g*hR)
    # wave speed estimates (Toro 2001) with dry-bed handling
    hs = (0.5*(cL+cR) + 0.25*(uL-uR))**2 / g
    us = 0.5*(uL+uR) + cL - cR
    SL = np.where(hL > HDRY, np.minimum(uL-cL, us-np.sqrt(g*hs)), uR-2*cR)
    SR = np.where(hR > HDRY, np.maximum(uR+cR, us+np.sqrt(g*hs)), uL+2*cL)
    FL = np.stack([hL*uL, hL*uL**2+0.5*g*hL**2]); FR = np.stack([hR*uR, hR*uR**2+0.5*g*hR**2])
    UL = np.stack([hL, hL*uL]); UR = np.stack([hR, hR*uR])
    Fm = (SR*FL - SL*FR + SL*SR*(UR-UL)) / np.where(SR-SL == 0, 1e-12, SR-SL)
    return np.where(SL >= 0, FL, np.where(SR <= 0, FR, Fm))

def minmod(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(abs(a), abs(b)), 0.0)

def rhs(U, dx):
    h, q = U
    Ue = np.concatenate([U[:, :1], U[:, :1], U, U[:, -1:], U[:, -1:]], axis=1)  # transmissive
    d1 = Ue[:, 1:-1]-Ue[:, :-2]; d2 = Ue[:, 2:]-Ue[:, 1:-1]
    s = minmod(d1, d2)
    Uc = Ue[:, 1:-1]
    UL = (Uc + 0.5*s)[:, :-1]; UR = (Uc - 0.5*s)[:, 1:]
    hL = np.maximum(UL[0], 0); hR = np.maximum(UR[0], 0)
    uL = np.where(hL > HDRY, UL[1]/np.maximum(hL, HDRY), 0); uR = np.where(hR > HDRY, UR[1]/np.maximum(hR, HDRY), 0)
    F = hll_flux(hL, uL, hR, uR)
    return -(F[:, 1:]-F[:, :-1])/dx

def solve(h0, N, L=1000.0, x0=500.0, T=20.0, cfl=0.45):
    dx = L/N; x = (np.arange(N)+0.5)*dx
    U = np.zeros((2, N)); U[0] = h0(x)
    t = 0.0
    while t < T:
        h = U[0]; u = np.where(h > HDRY, U[1]/np.maximum(h, HDRY), 0)
        smax = np.max(abs(u)+np.sqrt(g*np.maximum(h, 0)))
        dt = min(cfl*dx/smax, T-t)
        U1 = U + dt*rhs(U, dx); U1[0] = np.maximum(U1[0], 0)
        U = 0.5*(U + U1 + dt*rhs(U1, dx)); U[0] = np.maximum(U[0], 0)
        U[1] = np.where(U[0] > HDRY, U[1], 0)
        t += dt
    return x, U

def ritter(x, t, hl, x0):
    c0 = np.sqrt(g*hl); xi = (x-x0)/t
    h = np.where(xi <= -c0, hl, np.where(xi >= 2*c0, 0, (2*c0-xi)**2/(9*g)))
    return h

def stoker(x, t, hl, hr, x0):
    cl, cr = np.sqrt(g*hl), np.sqrt(g*hr)
    # middle state from Riemann invariant (rarefaction) + Rankine-Hugoniot (shock)
    def F(hm):
        um = 2*(cl-np.sqrt(g*hm))
        S = hm*um/(hm-hr)
        return (hm*um**2 + 0.5*g*hm**2) - (0.5*g*hr**2) - S*(hm*um)
    hm = brentq(F, hr*1.0001, hl*0.9999); cm = np.sqrt(g*hm); um = 2*(cl-cm); S = hm*um/(hm-hr)
    xi = (x-x0)/t
    h = np.where(xi <= -cl, hl, np.where(xi <= um-cm, (2*cl-xi)**2/(9*g), np.where(xi <= S, hm, hr)))
    return h, hm, S

if __name__ == "__main__":
    out = {"dry": [], "wet": []}
    hl, hr, x0, T = 10.0, 1.0, 500.0, 20.0
    for N in [100, 200, 400, 800, 1600, 3200]:
        x, U = solve(lambda x: np.where(x < x0, hl, 0.0), N, T=T)
        e = np.mean(abs(U[0]-ritter(x, T, hl, x0)))
        out["dry"].append([N, float(e)])
        x, U = solve(lambda x: np.where(x < x0, hl, hr), N, T=T)
        hs, hm, S = stoker(x, T, hl, hr, x0)
        e = np.mean(abs(U[0]-hs))
        out["wet"].append([N, float(e)])
        print(N, out["dry"][-1][1], out["wet"][-1][1])
    out["stoker_hm"] = hm; out["stoker_S"] = S
    # profiles at N=400 for figure
    for case, hR in [("dry", 0.0), ("wet", hr)]:
        x, U = solve(lambda x: np.where(x < x0, hl, hR), 400, T=T)
        out["prof_"+case] = {"x": x.tolist(), "h": U[0].tolist()}
    json.dump(out, open("fv1d.json", "w"))
    print("hm", hm, "S", S)
