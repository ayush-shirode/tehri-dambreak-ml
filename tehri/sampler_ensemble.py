"""Step 3-4: sample scenarios from ranges stored in the knowledge graph, then run the
physics chain (breach hydrograph -> 2D SWE on the Copernicus DEM) for each scenario."""
import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from rdflib import Graph
from scipy.stats import qmc

KG = "tehri_kg.ttl"
def kg_ranges():
    g = Graph().parse(KG)
    q = """PREFIX dbk:<http://example.org/dambreak#>
    SELECT ?n ?v ?mn ?mx ?st WHERE { ?p dbk:name ?n ; dbk:status ?st .
      ?o dbk:hasParameter ?p . FILTER(?o IN (dbk:TehriScenarioSpace, dbk:TehriDam, dbk:Froehlich2008Model))
      OPTIONAL{?p dbk:value ?v} OPTIONAL{?p dbk:min ?mn} OPTIONAL{?p dbk:max ?mx} }"""
    out = {}
    for n, v, mn, mx, st in g.query(q):
        out[str(n)] = dict(value=float(v) if v else None, min=float(mn) if mn else None, max=float(mx) if mx else None, status=str(st))
    return out

R = kg_ranges()
P_OVERTOP = 0.25   # design choice: share of overtopping scenarios in the ensemble (not a probability estimate)

def design(n, seed):
    u = qmc.LatinHypercube(d=5, seed=seed).random(n)
    lv = R["reservoirLevel"]; wf = R["breachWidthFactor"]; mn = R["manningN"]
    mode = (u[:, 1] < P_OVERTOP).astype(int)
    level = np.where(mode == 1, R["crestElevation"]["value"], lv["min"] + u[:, 0]*(lv["max"]-lv["min"]))
    return np.c_[level, mode, u[:, 2], wf["min"] + u[:, 3]*(wf["max"]-wf["min"]), mn["min"] + u[:, 4]*(mn["max"]-mn["min"])]

GK = [5, 10, 17, 25, 33, 38.8]   # chainages (km); 38.8 km = Devprayag on the DEM thalweg

def worker(args):
    k, prm = args
    import domain, breach, swe_dem
    z = domain.load(); path, ch = domain.thalweg(z)
    r0, c0 = path[0]; rr, cc = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    src = (np.hypot(rr-r0, cc-c0)*domain.DX <= 200) & (z < z[r0, c0]+15)
    st = breach.Storage(R["crestElevation"]["value"]-R["height"]["value"])
    level, mode, tf_frac, wfac, n = prm
    t, q, zr, p = breach.hydrograph(level, "overtopping" if mode > 0.5 else "piping", wfac, tf_frac, st,
                                    z_bed=z[r0, c0], z_crest=R["crestElevation"]["value"], T=6*3600)
    gauges = [domain.gauge(z, path, ch, km) for km in GK]
    t0 = time.time()
    r = swe_dem.run(z, domain.DX, src, t, q, n, 6*3600, gauges)
    newly = (r["hmax"] > 1.0) & ~(z.astype(np.float32) == 0)
    return dict(k=k, prm=list(map(float, prm)), Qp=float(q.max()), tQp=float(t[q.argmax()]), breach={kk: float(v) for kk, v in p.items()},
                gauge_peak=r["gauge_peak"].tolist(), gauge_arr=r["gauge_arr"].tolist(), area_km2=float(newly.sum()*domain.DX**2/1e6),
                vin=r["vin"], runtime=time.time()-t0, hmax=r["hmax"].astype(np.float32))

if __name__ == "__main__":
    ntr, nte = int(sys.argv[1]), int(sys.argv[2])
    P = np.vstack([design(ntr, 11), design(nte, 22)]); split = np.r_[np.zeros(ntr), np.ones(nte)].astype(int)
    json.dump({"ranges_from_kg": R, "P_overtop": P_OVERTOP}, open("ens_ranges.json", "w"), indent=1)
    res = []
    with Pool(2) as pool:
        for out in pool.imap_unordered(worker, list(enumerate(P))):
            res.append(out)
            print(len(res), out["k"], round(out["runtime"]), "Qp", round(out["Qp"]), "arrDev", round(out["gauge_arr"][-1]/60, 1), flush=True)
    res.sort(key=lambda d: d["k"])
    np.savez_compressed("tehri_ensemble.npz", P=P, split=split, H=np.array([d["hmax"] for d in res]),
                        peak=np.array([d["gauge_peak"] for d in res]), arr=np.array([d["gauge_arr"] for d in res]),
                        area=np.array([d["area_km2"] for d in res]), Qp=np.array([d["Qp"] for d in res]),
                        tQp=np.array([d["tQp"] for d in res]), rt=np.array([d["runtime"] for d in res]))
    json.dump([{k: v for k, v in d.items() if k != "hmax"} for d in res], open("tehri_ensemble_meta.json", "w"))
    print("done")
