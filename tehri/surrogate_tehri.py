"""Step 5: train/test surrogates on the Tehri ensemble.
Inputs A (raw):    level, mode, tf_frac, width factor, Manning n
Inputs B (hybrid): raw + outputs of the cheap breach model (log Qp, time of Qp, log Vw)
Targets: gauge peak depths, arrival times (log), flooded area, and the hmax field via POD."""
import numpy as np, json, time, warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = np.load("tehri_ensemble.npz")
meta = json.load(open("tehri_ensemble_meta.json"))
P, split, H, peak, arr, area, Qp, tQp, rt = (D[k] for k in ["P", "split", "H", "peak", "arr", "area", "Qp", "tQp", "rt"])
Vw = np.array([m["breach"]["Vw"] for m in meta]); Qf95 = np.array([m["breach"]["Qp_froehlich1995"] for m in meta])
tr, te = split == 0, split == 1
Hf = H.reshape(len(P), -1).astype(float)
wet_any = Hf.max(0) > 0.05; Hf = Hf[:, wet_any]            # keep cells wet in at least one scenario

def X(kind, idx):
    base = P[idx]
    if kind == "raw": return base
    return np.c_[base, np.log(Qp[idx]), tQp[idx]/3600, np.log(Vw[idx])]

def gpr(d):
    k = Ck(1.0)*RBF(np.ones(d), (1e-2, 1e3)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
def mlp(d):
    return MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=5000, alpha=1e-4,
                        learning_rate_init=3e-3, random_state=0, tol=1e-7)
def fit_pred(model, Xtr, Ytr, Xte, return_std=False):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(Ytr)
    m = model(Xtr.shape[1]); m.fit(sx.transform(Xtr), sy.transform(Ytr))
    if return_std:
        mu, sd = m.predict(sx.transform(Xte), return_std=True)
        sd = np.atleast_2d(sd.T).T if sd.ndim == 1 else sd
        return sy.inverse_transform(mu.reshape(len(Xte), -1)), sd*sy.scale_
    return sy.inverse_transform(m.predict(sx.transform(Xte)).reshape(len(Xte), -1))

res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "solver_runtime_mean_s": float(rt.mean()), "solver_runtime_std_s": float(rt.std()),
       "Qp_range": [float(Qp.min()), float(Qp.max())], "Qp_froehlich95_range": [float(Qf95.min()), float(Qf95.max())],
       "Qp_ratio_routed_over_F95": [float((Qp/Qf95).min()), float(np.median(Qp/Qf95)), float((Qp/Qf95).max())],
       "arr_dev_min": [float(np.nanmin(arr[:, -1])/60), float(np.nanmax(arr[:, -1])/60)], "nan_arr": int(np.isnan(arr).sum()),
       "peak_dev_m": [float(peak[:, -1].min()), float(peak[:, -1].max())], "area_km2": [float(area.min()), float(area.max())],
       "n_cells_field": int(Hf.shape[1])}
Y = np.c_[np.log(arr), peak, area]
def smet(Yp, Yt):
    ta, tp = np.exp(Yt[:, :6]), np.exp(Yp[:, :6])
    return {"arr_MAPE_%": float(100*np.mean(abs(tp-ta)/ta)), "arr_MAE_min": float(np.mean(abs(tp-ta))/60),
            "peak_MAE_m": float(np.mean(abs(Yp[:, 6:12]-Yt[:, 6:12]))), "peak_MAPE_%": float(100*np.mean(abs(Yp[:, 6:12]-Yt[:, 6:12])/Yt[:, 6:12])),
            "area_MAPE_%": float(100*np.mean(abs(Yp[:, 12]-Yt[:, 12])/Yt[:, 12]))}
res["scalar"] = {}; preds = {}
for kind in ["raw", "hybrid"]:
    for name, model in [("GPR", gpr), ("MLP", mlp)]:
        Yp = fit_pred(model, X(kind, tr), Y[tr], X(kind, te)); res["scalar"][f"{name}-{kind}"] = smet(Yp, Y[te]); preds[f"{name}-{kind}"] = Yp

mu = Hf[tr].mean(0); U, S, Vt = np.linalg.svd(Hf[tr]-mu, full_matrices=False)
e = np.cumsum(S**2)/np.sum(S**2); r = int(np.searchsorted(e, 0.999)+1)
res["pod_modes_99.9"] = r; res["pod_energy_first3"] = e[:3].tolist()
res["pod_projection_relL2_%"] = float(100*np.mean(np.linalg.norm((Hf[te]-mu)-(Hf[te]-mu)@Vt[:r].T@Vt[:r], axis=1)/np.linalg.norm(Hf[te], axis=1)))
def fmet(Hp, Ht, thr=1.0):
    rel = np.linalg.norm(Hp-Ht, axis=1)/np.linalg.norm(Ht, axis=1); A, B = Hp > thr, Ht > thr
    csi = (A & B).sum(1)/(A | B).sum(1)
    return {"relL2_%_mean": float(100*rel.mean()), "relL2_%_max": float(100*rel.max()), "CSI_mean": float(csi.mean()), "CSI_min": float(csi.min()),
            "MAE_wet_m": float(np.mean([abs(Hp[i]-Ht[i])[B[i]].mean() for i in range(len(Ht))]))}
res["field"] = {}; fp = {}
a_tr = (Hf[tr]-mu)@Vt[:r].T
for kind in ["raw", "hybrid"]:
    for name, model in [("POD-GPR", gpr), ("POD-NN", mlp)]:
        a_p = fit_pred(model, X(kind, tr), a_tr, X(kind, te)); Hp = np.maximum(mu+a_p@Vt[:r], 0)
        res["field"][f"{name}-{kind}"] = fmet(Hp, Hf[te]); fp[f"{name}-{kind}"] = Hp
# GPR predictive uncertainty for arrival at Devprayag (hybrid): coverage of ~95% interval on test set
mu_a, sd_a = fit_pred(gpr, X("hybrid", tr), np.log(arr[tr, -1:]), X("hybrid", te), return_std=True)
lo, hi = np.exp(mu_a[:, 0]-1.96*sd_a[:, 0]), np.exp(mu_a[:, 0]+1.96*sd_a[:, 0]); ta = arr[te, -1]
res["gpr_arrDev_95PI_coverage"] = float(np.mean((ta >= lo) & (ta <= hi))); res["gpr_arrDev_mean_PI_width_min"] = float(np.mean(hi-lo)/60)
# inference timing
sx = StandardScaler().fit(X("hybrid", tr)); sy = StandardScaler().fit(a_tr); m = gpr(X("hybrid", tr).shape[1]); m.fit(sx.transform(X("hybrid", tr)), sy.transform(a_tr))
Xq = sx.transform(X("hybrid", te)); t0 = time.time()
for _ in range(20): _ = mu + sy.inverse_transform(m.predict(Xq))@Vt[:r]
res["inference_s_per_scenario"] = (time.time()-t0)/(20*len(Xq))
res["speedup"] = res["solver_runtime_mean_s"]/res["inference_s_per_scenario"]
json.dump(res, open("res_tehri.json", "w"), indent=1)
np.savez_compressed("preds_tehri.npz", Hte=Hf[te], wet=wet_any, Yte=Y[te], Pte=P[te], lo=lo, hi=hi, ta=ta,
                    **{k.replace("-", "_"): v for k, v in fp.items()}, **{"s_"+k.replace("-", "_"): v for k, v in preds.items()})
print(json.dumps(res, indent=1))
