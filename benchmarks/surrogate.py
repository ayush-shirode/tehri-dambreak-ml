"""Experiment D: machine-learning surrogates trained on the FV dam-break ensemble.
(i) scalar hazard indicators (arrival time, peak depth at gauges, inundated area);
(ii) full max-depth field via POD + regression (POD-GPR / POD-NN).
Compares data-only inputs with physics-informed (log/scaling) inputs."""
import numpy as np, json, time, warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
g = 9.81
D = np.load("ensemble.npz")
P, split, Hf, arr, peak, area, rt, V0 = (D[k] for k in ["P", "split", "H", "arr", "peak", "area", "rt", "V0"])
tr, te = split == 0, split == 1
nx, ny = Hf.shape[1:]; Hf = Hf.reshape(len(P), -1).astype(float)

def feats(kind, idx):
    Hd, n, S0 = P[idx].T
    if kind == "raw":
        return np.c_[Hd, n, S0]
    # physics-informed: log of governing scales (power-law structure of SWE + Manning)
    return np.c_[np.log(Hd), np.log(n), np.log(S0), np.log(V0[idx]), np.log(np.sqrt(g*Hd))]

def gpr(d):
    k = Ck(1.0)*RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
def mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=5000, alpha=1e-4, learning_rate_init=3e-3,
                        random_state=0, tol=1e-7)

def fit_pred(model, Xtr, Ytr, Xte):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(Ytr)
    m = model(Xtr.shape[1]) if model is gpr else model()
    m.fit(sx.transform(Xtr), sy.transform(Ytr))
    return sy.inverse_transform(m.predict(sx.transform(Xte)).reshape(len(Xte), -1))

res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "solver_runtime_mean_s": float(rt.mean()),
       "solver_runtime_std_s": float(rt.std()), "param_ranges": {"Hd": [float(P[:,0].min()), float(P[:,0].max())],
       "n": [float(P[:,1].min()), float(P[:,1].max())], "S0": [float(P[:,2].min()), float(P[:,2].max())]},
       "V0_Mm3": [float(V0.min()/1e6), float(V0.max()/1e6)], "area_km2": [float(area.min()), float(area.max())],
       "arr_s_range": [float(np.nanmin(arr)), float(np.nanmax(arr))], "nan_arrivals": int(np.isnan(arr).sum())}

# ---------- (i) scalar indicators ----------
Y = np.c_[np.log(arr), peak, area]           # 4 arrival times (log), 4 peak depths, area
names = [f"t_arr_{k}" for k in (4, 6, 8, 10)] + [f"h_pk_{k}" for k in (4, 6, 8, 10)] + ["area"]
def scalar_metrics(Yp, Yt):
    out = {}
    ta, tp = np.exp(Yt[:, :4]), np.exp(Yp[:, :4])
    out["arr_MAPE_%"] = float(100*np.mean(abs(tp-ta)/ta))
    out["peak_MAE_m"] = float(np.mean(abs(Yp[:, 4:8]-Yt[:, 4:8])))
    out["peak_MAPE_%"] = float(100*np.mean(abs(Yp[:, 4:8]-Yt[:, 4:8])/Yt[:, 4:8]))
    out["area_MAPE_%"] = float(100*np.mean(abs(Yp[:, 8]-Yt[:, 8])/Yt[:, 8]))
    ss = ((Yp-Yt)**2).sum(0); st = ((Yt-Yt.mean(0))**2).sum(0)
    out["R2_mean"] = float(np.mean(1-ss/st))
    return out
res["scalar"] = {}
preds = {}
for fk in ["raw", "phys"]:
    for mname, model in [("GPR", gpr), ("MLP", mlp)]:
        t0 = time.time()
        Yp = fit_pred(model, feats(fk, tr), Y[tr], feats(fk, te))
        res["scalar"][f"{mname}-{fk}"] = scalar_metrics(Yp, Y[te]); preds[f"{mname}-{fk}"] = Yp
        res["scalar"][f"{mname}-{fk}"]["fit_s"] = time.time()-t0
# pure physics-scaling baseline: multivariate power law (linear in log space)
Xl = feats("phys", tr)[:, :4]; Xlt = feats("phys", te)[:, :4]
Yl = np.c_[np.log(arr), np.log(peak), np.log(area)][tr]
lr = LinearRegression().fit(Xl, Yl); Ypl = lr.predict(Xlt)
Ypl = np.c_[Ypl[:, :4], np.exp(Ypl[:, 4:])]
res["scalar"]["PowerLaw-phys"] = scalar_metrics(Ypl, Y[te]); preds["PowerLaw-phys"] = Ypl
res["powerlaw_arr_exponents_[Hd,n,S0,V0]"] = lr.coef_[:4].mean(0).tolist()

# ---------- (ii) field surrogate: POD + regression ----------
mu = Hf[tr].mean(0)
Uu, S, Vt = np.linalg.svd(Hf[tr]-mu, full_matrices=False)
energy = np.cumsum(S**2)/np.sum(S**2)
r = int(np.searchsorted(energy, 0.999)+1)
res["pod_modes_99.9"] = r; res["pod_energy_first5"] = energy[:5].tolist()
proj_err = np.linalg.norm((Hf[te]-mu) - (Hf[te]-mu)@Vt[:r].T@Vt[:r], axis=1)/np.linalg.norm(Hf[te], axis=1)
res["pod_projection_relL2_%"] = float(100*proj_err.mean())
def field_metrics(Hp, Ht, thr=0.3):
    rel = np.linalg.norm(Hp-Ht, axis=1)/np.linalg.norm(Ht, axis=1)
    A, B = Hp > thr, Ht > thr
    csi = (A & B).sum(1)/((A | B).sum(1))
    return {"relL2_%_mean": float(100*rel.mean()), "relL2_%_max": float(100*rel.max()),
            "CSI_mean": float(csi.mean()), "CSI_min": float(csi.min()),
            "MAE_wet_m": float(np.mean([abs(Hp[i]-Ht[i])[B[i]].mean() for i in range(len(Ht))]))}
res["field"] = {}; fpred = {}
for fk in ["raw", "phys"]:
    for mname, model in [("POD-GPR", gpr), ("POD-NN", mlp)]:
        a_tr = (Hf[tr]-mu)@Vt[:r].T
        t0 = time.time()
        a_p = fit_pred(model, feats(fk, tr), a_tr, feats(fk, te))
        Hp = np.maximum(mu + a_p@Vt[:r], 0)
        res["field"][f"{mname}-{fk}"] = field_metrics(Hp, Hf[te]); fpred[f"{mname}-{fk}"] = Hp
# inference timing for best field model
sx = StandardScaler().fit(feats("phys", tr)); a_tr = (Hf[tr]-mu)@Vt[:r].T; sy = StandardScaler().fit(a_tr)
m = gpr(5); m.fit(sx.transform(feats("phys", tr)), sy.transform(a_tr))
Xq = sx.transform(feats("phys", te)); t0 = time.time()
for _ in range(20): Hq = mu + sy.inverse_transform(m.predict(Xq))@Vt[:r]
res["podgpr_inference_s_per_scenario"] = (time.time()-t0)/(20*len(Xq))
res["speedup_vs_solver"] = res["solver_runtime_mean_s"]/res["podgpr_inference_s_per_scenario"]

# ---------- learning curve ----------
lc = {}
idx_tr = np.where(tr)[0]
for ntr in [30, 60, 120, 240]:
    sub = np.zeros(len(P), bool); sub[idx_tr[:ntr]] = True
    row = {}
    for fk in ["raw", "phys"]:
        mu_s = Hf[sub].mean(0); _, _, V_s = np.linalg.svd(Hf[sub]-mu_s, full_matrices=False)
        a_s = (Hf[sub]-mu_s)@V_s[:r].T
        a_p = fit_pred(gpr, feats(fk, sub), a_s, feats(fk, te))
        row[f"field_relL2_{fk}"] = field_metrics(np.maximum(mu_s + a_p@V_s[:r], 0), Hf[te])["relL2_%_mean"]
        Yp = fit_pred(gpr, feats(fk, sub), Y[sub], feats(fk, te))
        row[f"arr_MAPE_{fk}"] = scalar_metrics(Yp, Y[te])["arr_MAPE_%"]
    lc[ntr] = row
res["learning_curve_GPR"] = lc
json.dump(res, open("res_d.json", "w"), indent=1)
np.savez_compressed("preds.npz", Yte=Y[te], Pte=P[te], Hte=Hf[te],
                    **{k.replace("-", "_"): v for k, v in fpred.items()}, **{"s_"+k.replace("-", "_"): v for k, v in preds.items()})
print(json.dumps(res, indent=1))
