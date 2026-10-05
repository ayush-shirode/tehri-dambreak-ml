"""Model domain for Tehri -> Devprayag on the Copernicus GLO-30 DEM, resampled to 90 m UTM 44N."""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from pyproj import Transformer

DX = 90.0
T60 = Affine(*eval(open("grid_60m_transform.txt").read()))
# bounds taken from the 60 m overview grid: rows 152..612, cols 380..712 (dam toe to below Devprayag)
x0, y0 = T60*(380, 152); x1, y1 = T60*(712, 612)
NX, NY = int((y0-y1)//DX), int((x1-x0)//DX)          # NX rows (north->south), NY cols
TR = Affine(DX, 0, x0, 0, -DX, y0)
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
to_ll = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)

def load():
    z = np.empty((NX, NY), np.float32)
    with rasterio.open("dem_N30E078.tif") as d:
        reproject(rasterio.band(d, 1), z, dst_transform=TR, dst_crs="EPSG:32644", resampling=Resampling.average)
    z = z.astype(float)
    z[0, :] = 900.0   # closed northern boundary = downstream face of Tehri dam (crest 839.5 m)
    return z

def rc(lon, lat):
    x, y = to_utm.transform(lon, lat); c, r = ~TR*(x, y); return int(r), int(c)

def thalweg(z):
    """Steepest-descent path on a priority-flood-filled DEM from the lowest cell near the dam toe."""
    import heapq
    H, W = z.shape; f = z.copy(); seen = np.zeros(z.shape, bool); pq = []
    for r in range(H):
        for c in range(W):
            if r == H-1 or ((c in (0, W-1)) and r > H//2): heapq.heappush(pq, (f[r, c], r, c)); seen[r, c] = True
    while pq:
        e, r, c = heapq.heappop(pq)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc]:
                    seen[rr, cc] = True; f[rr, cc] = max(z[rr, cc], e+1e-3); heapq.heappush(pq, (f[rr, cc], rr, cc))
    r0, c0 = rc(78.4808, 30.3778)
    win = z[1:6, c0-6:c0+7]; k = np.unravel_index(np.argmin(win), win.shape)
    r, c = 1+k[0], c0-6+k[1]; path = [(r, c)]
    while r < H-1 and 0 < c < W-1:
        nb = np.array([[(f[r+a, c+b]-f[r, c])/np.hypot(a, b) if (a or b) else 0 for b in (-1, 0, 1)] for a in (-1, 0, 1)])
        k = np.unravel_index(np.argmin(nb), nb.shape)
        if nb[k] >= 0: break
        r, c = r+k[0]-1, c+k[1]-1; path.append((r, c))
    path = np.array(path); ch = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))*DX]/1000
    return path, ch

def gauge(z, path, ch, km, radius=270.0):
    """Gauge = cells within `radius` of the thalweg point at chainage km (captures the channel)."""
    i = min(np.searchsorted(ch, km), len(path)-1); r, c = path[i]
    rr, cc = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    return np.where(np.hypot(rr-r, cc-c)*DX <= radius)
