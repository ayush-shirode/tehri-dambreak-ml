import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import domain, breach
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
D = np.load("tehri_ensemble.npz"); meta = json.load(open("tehri_ensemble_meta.json"))
p = np.load("preds_tehri.npz"); R = json.load(open("res_tehri.json"))
z = domain.load(); path, ch = domain.thalweg(z); wet = p["wet"]
GK = [5, 10, 17, 25, 33, 38.8]
out = {}
# ---------- Fig 6: map + hydrographs ----------
Hte = p["Hte"]; Hp = p["POD_GPR_hybrid"]
err = np.linalg.norm(Hp-Hte, axis=1)/np.linalg.norm(Hte, axis=1); k = int(np.argsort(err)[len(err)//2])
out["fig6_case"] = p["Pte"][k].tolist(); out["fig6_relL2_%"] = float(100*err[k])
full = np.full(z.size, np.nan); full[wet] = Hte[k]; hm = full.reshape(z.shape)
fig = plt.figure(figsize=(7.0, 3.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
ax = fig.add_subplot(gs[0]); ext = [0, z.shape[1]*domain.DX/1000, z.shape[0]*domain.DX/1000, 0]
gy, gx = np.gradient(z, domain.DX); hs = np.clip(0.55+0.45*(gx-gy)/np.hypot(1, np.hypot(gx, gy))*2, 0, 1)
ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.6, vmax=1.3)
im = ax.imshow(np.ma.masked_less(hm, 1.0), cmap="Blues", extent=ext, vmin=0, vmax=np.nanpercentile(hm[hm > 1], 99))
ax.plot(path[:, 1]*0.09, path[:, 0]*0.09, color=C2, lw=0.6)
for km in GK:
    i = np.searchsorted(ch, km); r, c = path[min(i, len(path)-1)]
    ax.plot(c*0.09, r*0.09, "o", ms=3.5, mfc="white", mec="#222222", mew=0.8)
    lab = {17: "17 km (Koteshwar)", 38.8: "Devprayag"}.get(km, f"{km:g} km")
    ax.annotate(lab, (c*0.09, r*0.09), xytext=(4, -2), textcoords="offset points", fontsize=6.5, color="#111111", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
ax.plot(path[0, 1]*0.09, path[0, 0]*0.09, "^", ms=6, color="#222222"); ax.annotate("Tehri Dam", (path[0, 1]*0.09, path[0, 0]*0.09), xytext=(5, -6), textcoords="offset points", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
ax.set_xlabel("Easting (km, local)"); ax.set_ylabel("Northing (km, local, downward)")
ax.set_title("(a) h$_{max}$ (m), median test scenario", fontsize=8.5, loc="left")
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
ax2 = fig.add_subplot(gs[1])
st = breach.Storage(839.5-260.5)
r0, c0 = path[0]
for (lev, mode, tff, wf, lab, c, ls) in [(835, "piping", 0.0, 1.0, "piping, MWL, Froehlich t$_f$", C2, "-"),
                                         (835, "piping", 1.0, 1.0, "piping, MWL, Von Thun t$_f$", C1, "--"),
                                         (740, "piping", 0.5, 1.0, "piping, MDDL, mid t$_f$", C3, ":")]:
    t, q, zr, pp = breach.hydrograph(lev, mode, wf, tff, st, z_bed=z[r0, c0], T=6*3600)
    ax2.plot(t/3600, q/1e3, color=c, ls=ls, lw=1.4, label=lab)
    ax2.axhline(pp["Qp_froehlich1995"]/1e3, color=c, ls=ls, lw=0.6, alpha=0.7)
ax2.set_xlabel("Time after breach start (h)"); ax2.set_ylabel("Breach outflow (10$^3$ m$^3$/s)")
ax2.set_title("(b) Breach hydrographs", fontsize=8.5, loc="left")
ax2.plot([], [], color="#777777", lw=0.6, label="thin lines: Froehlich (1995) Q$_p$")
ax2.legend(fontsize=6.5, frameon=False); ax2.grid(True, color="#e6e6e6", lw=0.6)
fig.tight_layout(); fig.savefig("fig6_tehri.png", dpi=300)

# ---------- Fig 7: surrogate on Tehri ----------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
ta = p["ta"]/60; lo, hi = p["lo"]/60, p["hi"]/60; pr = np.sqrt(lo*hi)
ax[0].errorbar(ta, pr, yerr=[pr-lo, hi-pr], fmt="o", ms=3, mfc="none", color=C1, ecolor="#9cc2ee", elinewidth=0.8, capsize=0)
l0, l1 = ta.min()*0.95, ta.max()*1.05; ax[0].plot([l0, l1], [l0, l1], color="#333333", lw=0.8)
ax[0].set_xlabel("Solver arrival (min)"); ax[0].set_ylabel("GPR prediction (min)")
ax[0].set_title("(a) Arrival at Devprayag, 95% PI", fontsize=8.5)
names = ["GPR-raw", "GPR-hybrid", "MLP-raw", "MLP-hybrid"]
v = [R["scalar"][n]["peak_MAE_m"] for n in names]
ax[1].bar(range(4), v, color=[C1, C1, C2, C2], width=0.6)
for i in (1, 3): ax[1].patches[i].set_hatch("////"); ax[1].patches[i].set_edgecolor("white")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["GPR\nraw", "GPR\nhybrid", "MLP\nraw", "MLP\nhybrid"], fontsize=7)
ax[1].set_ylabel("Peak-depth MAE (m)"); ax[1].set_title("(b) Gauge peak depth error", fontsize=8.5)
for i, val in enumerate(v): ax[1].text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
fn = ["POD-GPR-raw", "POD-GPR-hybrid", "POD-NN-raw", "POD-NN-hybrid"]
v2 = [R["field"][n]["relL2_%_mean"] for n in fn]
ax[2].bar(range(4), v2, color=[C1, C1, C2, C2], width=0.6)
for i in (1, 3): ax[2].patches[i].set_hatch("////"); ax[2].patches[i].set_edgecolor("white")
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["GPR\nraw", "GPR\nhybrid", "NN\nraw", "NN\nhybrid"], fontsize=7)
ax[2].set_ylabel("Rel. L$_2$ error of h$_{max}$ (%)"); ax[2].set_title("(c) POD flood-map error", fontsize=8.5)
for i, val in enumerate(v2): ax[2].text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
for a in ax: a.grid(True, axis="y", color="#e6e6e6", lw=0.6)
fig.tight_layout(); fig.savefig("fig7_tehri_ml.png", dpi=300)
json.dump(out, open("res_fig_tehri.json", "w")); print(out)
