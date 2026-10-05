import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fv1d
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
g = 9.81
res = {}

# ---------- Fig 2: verification ----------
d = json.load(open("fv1d.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
hl, hr, x0, T = 10.0, 1.0, 500.0, 20.0
xa = np.linspace(0, 1000, 2000)
for k, (case, hR) in enumerate([("dry", 0.0), ("wet", hr)]):
    p = d["prof_"+case]
    ex = fv1d.ritter(xa, T, hl, x0) if case == "dry" else fv1d.stoker(xa, T, hl, hr, x0)[0]
    ax[k].plot(xa, ex, color="#333333", lw=1.2, label="Analytical")
    ax[k].plot(p["x"][::8], p["h"][::8], "o", ms=3, mfc="none", mew=0.8, color=C1, label="FV (N=400)", zorder=3)
    ax[k].set_title(("(a) Dry bed vs Ritter" if case == "dry" else "(b) Wet bed vs Stoker"), fontsize=9)
    ax[k].set_xlabel("x (m)"); ax[k].set_ylabel("h (m)"); ax[k].legend(fontsize=7, frameon=False, loc="lower left")
N = np.array([r[0] for r in d["dry"]])
for case, c, mk, lab in [("dry", C1, "o-", "Dry bed"), ("wet", C2, "s--", "Wet bed")]:
    e = np.array([r[1] for r in d[case]])
    ax[2].loglog(1000/N, e, mk, color=c, ms=4, lw=1.2, label=lab)
    res["order_"+case] = float(np.polyfit(np.log(1000/N), np.log(e), 1)[0])
    res["L1_"+case] = d[case]
ax[2].loglog(1000/N, 0.05*(1000/N/10), ":", color="#777777", lw=1, label="slope 1")
ax[2].set_xlabel(r"$\Delta x$ (m)"); ax[2].set_ylabel(r"L$_1$ error in h (m)"); ax[2].set_title("(c) Grid convergence", fontsize=9)
ax[2].legend(fontsize=7, frameon=False, loc="upper left")
ax[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); fig.savefig("fig2_verif.png", dpi=300)
res["stoker_hm"] = d["stoker_hm"]; res["stoker_S"] = d["stoker_S"]

# ---------- Fig 3: SPH vs FV-SWE vs Ritter ----------
H, A = 1.0, 0.5
# symmetric FV problem = column against a reflecting wall
L = 14.0; xc = 7.0; Ncell = 2800
fv_front = []
for Ts in np.arange(0.25, 2.51, 0.25):
    t = Ts*np.sqrt(H/g)
    x, U = fv1d.solve(lambda x: np.where(abs(x-xc) < A, H, 0.0), Ncell, L=L, x0=xc, T=t)
    wet = x[(U[0] > 0.01) & (x > xc)]
    fv_front.append([Ts, (wet.max()-xc)/A])
fv_front = np.array(fv_front)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
Ts = np.linspace(0, 2.5, 50)
ax[0].plot(Ts, 1+2*np.sqrt(g*H)*Ts*np.sqrt(H/g)/A, color="#333333", lw=1.2, label="Ritter (SWE, exact)")
ax[0].plot(fv_front[:, 0], fv_front[:, 1], "s--", color=C2, ms=4, lw=1.2, label="FV-SWE (h > 1 cm)")
for dp, c, mk in [(0.02, C3, ":"), (0.01, C1, "-")]:
    f = np.array(json.load(open(f"sph_{dp}.json"))["front"])
    ax[0].plot(f[:, 0], f[:, 1], mk, color=c, lw=1.6, label=f"WCSPH, $\\Delta p$={dp} m")
    res[f"sph_front_{dp}"] = [[float(a), float(b)] for a, b in f[np.searchsorted(f[:, 0], [1.0, 1.5, 2.0, 2.4])]]
res["fv_front"] = fv_front.tolist()
ax[0].set_xlabel(r"$T^*=t\sqrt{g/H}$"); ax[0].set_ylabel(r"Front position $x_f/a$")
ax[0].set_title("(a) Surge-front position", fontsize=9); ax[0].legend(fontsize=7, frameon=False)
s = json.load(open("sph_0.01.json"))
key = sorted(s["snaps"].keys(), key=float)[2]
sn = s["snaps"][key]
sc = ax[1].scatter(sn["x"], sn["y"], c=sn["v"], s=0.6, cmap="Blues", vmin=0, vmax=max(sn["v"]))
ax[1].grid(False); ax[1].set_aspect("equal"); ax[1].set_xlim(0, 2.4); ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)"); ax[1].set_title(f"(b) WCSPH particles, $T^*$={key}", fontsize=9)
cb = fig.colorbar(sc, ax=ax[1], fraction=0.035, pad=0.02); cb.set_label("|v| (m/s)", fontsize=8)
fig.tight_layout(); fig.savefig("fig3_sph.png", dpi=300)
res["sph_nf"] = {"0.02": json.load(open("sph_0.02.json"))["nf"], "0.01": s["nf"]}
json.dump(res, open("res_ab.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:3000])
