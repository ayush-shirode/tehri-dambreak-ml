import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fv2d
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
r = json.load(open("res_d.json")); p = np.load("preds.npz")
D = np.load("ensemble.npz")
Hte = p["Hte"]; Hp = p["POD_GPR_phys"]; Pte = p["Pte"]
nx = 250; ny = 60
# pick the test scenario with the median POD-GPR error (representative, not cherry-picked)
err = np.linalg.norm(Hp-Hte, axis=1)/np.linalg.norm(Hte, axis=1)
k = int(np.argsort(err)[len(err)//2]); res = {"fig4_case": Pte[k].tolist(), "fig4_relL2_%": float(100*err[k])}
X, Y, z = fv2d.make_valley(Pte[k, 2]); X = X[50:]/1000; Y = Y[50:]/1000; z = z[50:]
ht = Hte[k].reshape(nx, ny); hp = Hp[k].reshape(nx, ny)
fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.5), sharex=True)
ax = [ax[0], None, ax[1]]
vmax = np.percentile(ht[ht > 0.3], 99)
for a, f, t in [(ax[0], ht, "(a) FV-SWE solver: maximum depth h$_{max}$")]:
    a.contour(X, Y, z, levels=15, colors="#bbbbbb", linewidths=0.4)
    im = a.pcolormesh(X, Y, np.ma.masked_less(f, 0.3), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
    a.set_title(t, fontsize=9, loc="left"); a.set_ylabel("y (km)"); a.set_aspect("equal")
    fig.colorbar(im, ax=a, fraction=0.02, pad=0.01).set_label("h$_{max}$ (m)", fontsize=8)
d = hp-ht; lim = np.percentile(abs(d[(ht > 0.3) | (hp > 0.3)]), 99)
im = ax[2].pcolormesh(X, Y, np.ma.masked_where((ht < 0.3) & (hp < 0.3), d), cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
ax[2].contour(X, Y, ht, levels=[0.3], colors="#333333", linewidths=0.8)
ax[2].contour(X, Y, hp, levels=[0.3], colors=C2, linewidths=0.8, linestyles="--")
ax[2].set_title("(b) POD-GPR surrogate minus solver; 0.3 m extent: solver solid, surrogate dashed", fontsize=9, loc="left")
ax[2].set_xlabel("Distance along valley x (km)"); ax[2].set_ylabel("y (km)"); ax[2].set_aspect("equal")
fig.colorbar(im, ax=ax[2], fraction=0.02, pad=0.01).set_label("$\\Delta h$ (m)", fontsize=8)
fig.tight_layout(); fig.savefig("fig4_maps.png", dpi=300)

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
ta = np.exp(p["Yte"][:, 3])/60
for key, c, mk, lab in [("s_PowerLaw_phys", C2, "s", "Power law"), ("s_GPR_phys", C1, "o", "GPR")]:
    ax[0].plot(ta, np.exp(p[key][:, 3])/60, mk, color=c, ms=3.5, mfc="none", mew=0.8, label=lab)
lo, hi = ta.min()*0.95, ta.max()*1.05
ax[0].plot([lo, hi], [lo, hi], color="#333333", lw=0.8); ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
ax[0].set_xlabel("Solver arrival (min)"); ax[0].set_ylabel("Predicted (min)"); ax[0].set_title("(a) Arrival at 10 km", fontsize=9)
ax[0].legend(fontsize=7, frameon=False)
lc = r["learning_curve_GPR"]; ns = [int(n) for n in lc]
ax[1].plot(ns, [lc[str(n)]["field_relL2_raw"] for n in ns], "o-", color=C1, ms=4, label="Raw inputs")
ax[1].plot(ns, [lc[str(n)]["field_relL2_phys"] for n in ns], "s--", color=C2, ms=4, label="Physics features")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].minorticks_off(); ax[1].set_xticks(ns); ax[1].set_xticklabels(ns); ax[1].set_yticks([2,3,5,10]); ax[1].set_yticklabels(["2","3","5","10"])
ax[1].set_xlabel("Training scenarios"); ax[1].set_ylabel("Rel. L$_2$ error of h$_{max}$ (%)")
ax[1].set_title("(b) POD-GPR learning curve", fontsize=9); ax[1].legend(fontsize=7, frameon=False)
Hf = D["H"].reshape(len(D["P"]), -1).astype(float)[D["split"] == 0]
S = np.linalg.svd(Hf-Hf.mean(0), compute_uv=False); e = 1-np.cumsum(S**2)/np.sum(S**2)
ax[2].semilogy(np.arange(1, 31), e[:30], "o-", color=C1, ms=3)
ax[2].axhline(1e-3, color="#777777", ls=":", lw=1); ax[2].axvline(r["pod_modes_99.9"], color="#777777", ls=":", lw=1)
ax[2].set_xlabel("Number of POD modes r"); ax[2].set_ylabel("Unexplained energy")
ax[2].set_title("(c) POD spectrum", fontsize=9)
for a in ax: a.grid(True, color="#e6e6e6", lw=0.6)
fig.tight_layout(); fig.savefig("fig5_ml.png", dpi=300)
json.dump(res, open("res_fig.json", "w")); print(res)
