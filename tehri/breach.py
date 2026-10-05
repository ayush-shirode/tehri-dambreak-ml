"""Breach outflow hydrograph for Tehri: level-pool reservoir routing + gradually
growing trapezoidal breach with critical-flow (broad-crested weir) discharge.

Reservoir:  dV/dt = -Q_b(t)  (inflow neglected over the few-hour event; PMF << breach flow)
Breach:     bottom elevation and bottom width grow linearly to final values over t_f
Discharge:  critical flow through a trapezoid for upstream energy head H = z_w - z_b,
            Q = A_c * sqrt(g A_c / T_c), with H = y_c + A_c/(2 T_c)   (C_d = 1, losses neglected)
Storage:    V(z) = V_dead * ((z - z0)/(z_MDDL - z0))^m, fitted to the two sourced storages
            (dead storage at MDDL, gross storage at FRL); z0 = valley-floor elevation.
"""
import numpy as np
from scipy.optimize import brentq
g = 9.81

class Storage:
    def __init__(self, z0, z_mddl=740.0, z_frl=830.0, v_dead=925e6, v_gross=3540e6):
        self.z0 = z0
        self.m = np.log(v_gross/v_dead)/np.log((z_frl-z0)/(z_mddl-z0))
        self.a = v_dead/(z_mddl-z0)**self.m
    def V(self, z): return self.a*np.maximum(z-self.z0, 0)**self.m
    def z(self, V): return self.z0 + (V/self.a)**(1/self.m)

def critical_q(H, b, zs):
    """Critical discharge of a trapezoid (bottom b, side slope zs H:V) for energy head H."""
    if H <= 0: return 0.0
    f = lambda y: y + (b*y+zs*y*y)/(2*(b+2*zs*y)) - H
    yc = brentq(f, 1e-9, H)
    A = b*yc+zs*yc*yc; T = b+2*zs*yc
    return A*np.sqrt(g*A/T) if T > 0 else 0.0

def froehlich2008(Vw, hb, mode):
    K0 = 1.3 if mode == "overtopping" else 1.0
    zs = 1.0 if mode == "overtopping" else 0.7
    B = 0.27*K0*Vw**0.32*hb**0.04
    tf = 63.2*np.sqrt(Vw/(g*hb**2))
    return B, tf, zs

def vonthun_tf(hw):  # erosion-resistant, hours -> s
    return (0.02*hw+0.25)*3600.0

def froehlich1995_qp(Vw, hw):
    return 0.607*Vw**0.295*hw**1.24

def hydrograph(z_w0, mode, width_factor, tf_frac, storage, z_bed, z_crest=839.5, dt=5.0, T=6*3600):
    """Returns times, Q(t), reservoir level, and the breach parameters used.
    tf_frac in [0,1] interpolates breach time between Froehlich (0) and Von Thun-Gillette (1)."""
    hb = z_w0 - z_bed
    Vw = storage.V(z_w0)
    B, tf_fr, zs = froehlich2008(Vw, hb, mode)
    tf_vt = vonthun_tf(hb)
    tf = tf_fr + tf_frac*(tf_vt-tf_fr)
    B *= width_factor
    b_final = max(B - zs*hb, 0.0)            # bottom width from average width of a trapezoid
    z_top = z_crest if mode == "overtopping" else z_w0
    t, V, out_t, out_q, out_z = 0.0, Vw, [], [], []
    while t <= T:
        frac = min(t/tf, 1.0)
        zb = z_top - frac*(z_top-z_bed); bb = frac*b_final
        zw = storage.z(V)
        q = critical_q(zw-zb, bb, zs) if zw > zb else 0.0
        q = min(q, V/dt)
        out_t.append(t); out_q.append(q); out_z.append(zw)
        V -= q*dt; t += dt
    return (np.array(out_t), np.array(out_q), np.array(out_z),
            dict(Vw=Vw, hb=hb, B_avg=B, b_final=b_final, zs=zs, tf=tf, tf_froehlich=tf_fr, tf_vonthun=tf_vt,
                 Qp_froehlich1995=froehlich1995_qp(Vw, hb)))

if __name__ == "__main__":
    # provisional z0 / z_bed from sourced crest elevation minus height (foundation); replaced by DEM riverbed later
    z0 = 839.5-260.5
    st = Storage(z0)
    print("storage exponent m =", round(st.m, 3), " V(FRL) =", st.V(830)/1e6, "MCM  V(MDDL) =", st.V(740)/1e6)
    for zw, mode in [(830, "piping"), (740, "piping"), (839.5, "overtopping")]:
        t, q, zr, p = hydrograph(zw, mode, 1.0, 0.0, st, z0)
        print(mode, zw, "Qp=%.0f m3/s at t=%.2f h | Froehlich1995 Qp=%.0f | B=%.0f m tf=%.2f h (VTG %.2f h)" %
              (q.max(), t[q.argmax()]/3600, p["Qp_froehlich1995"], p["B_avg"], p["tf"]/3600, p["tf_vonthun"]/3600))
