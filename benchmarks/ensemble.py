"""Generate the dam-break scenario ensemble (Latin hypercube over reservoir depth,
Manning roughness and valley slope) with the 2D FV solver."""
import numpy as np, time
from scipy.stats import qmc
import fv2d

lo = np.array([15.0, 0.025, 0.001]); hi = np.array([45.0, 0.070, 0.004])
def sample(n, seed):
    return qmc.scale(qmc.LatinHypercube(d=3, seed=seed).random(n), lo, hi)
P = np.vstack([sample(240, 1), sample(60, 2)])
split = np.array([0]*240 + [1]*60)
H, A, PK, AREA, RT, V0 = [], [], [], [], [], []
for k, (Hd, n, S0) in enumerate(P):
    t0 = time.time(); r = fv2d.run(Hd, n, S0); RT.append(time.time()-t0)
    ds = r["X"][:, 0] >= r["xdam"]
    H.append(r["hmax"][ds].astype(np.float32))
    A.append([np.nanmin(r["arr"][i]) for i in r["gi"]])
    PK.append([r["hmax"][i].max() for i in r["gi"]])
    AREA.append(((r["hmax"] > 0.3) & (r["X"] >= r["xdam"])).sum()*40*40/1e6)
    V0.append(r["V0"])
    if k % 20 == 0: print(k, RT[-1], A[-1], PK[-1], AREA[-1], flush=True)
np.savez_compressed("ensemble.npz", P=P, split=split, H=np.array(H), arr=np.array(A),
                    peak=np.array(PK), area=np.array(AREA), rt=np.array(RT), V0=np.array(V0))
print("done mean runtime", np.mean(RT))
