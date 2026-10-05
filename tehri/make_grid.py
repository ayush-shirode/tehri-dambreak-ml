"""Step 0: define the 60 m UTM-44N reference grid from the Copernicus tile.
domain.py uses this transform to place the 90 m model domain (dam toe to below Devprayag).
Writes grid_60m_transform.txt."""
import rasterio
from rasterio.warp import calculate_default_transform
from rasterio.windows import from_bounds

with rasterio.open("dem_N30E078.tif") as d:
    w = from_bounds(78.20, 30.00, 78.70, 30.45, d.transform)
    a = d.read(1, window=w)
    T, W, H = calculate_default_transform(d.crs, "EPSG:32644", a.shape[1], a.shape[0],
                                          *rasterio.windows.bounds(w, d.transform), resolution=60)
open("grid_60m_transform.txt", "w").write(repr(tuple(T)[:6]))
print("grid", H, "x", W, "transform", tuple(T)[:6])
