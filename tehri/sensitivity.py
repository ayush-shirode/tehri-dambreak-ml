"""Spearman rank correlation of ensemble outputs (Devprayag gauge) with each sampled input."""
import numpy as np, json
from scipy.stats import spearmanr
D = np.load("tehri_ensemble.npz"); P = D["P"]
names = ["level", "mode", "tf_frac", "width", "n"]
out = {nm: {names[i]: round(spearmanr(P[:, i], y)[0], 2) for i in range(5)}
       for nm, y in [("arrDev", D["arr"][:, -1]/60), ("peakDev", D["peak"][:, -1]), ("Qp", D["Qp"])]}
json.dump(out, open("spearman.json", "w")); print(json.dumps(out))
