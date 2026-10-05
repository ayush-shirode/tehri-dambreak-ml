import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8})
fig, ax = plt.subplots(figsize=(7.0, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=c, lw=1.3))
    ax.text(x+w/2, y+h-2.2, title, ha="center", va="top", weight="bold", fontsize=8, color="#222222")
    ax.text(x+w/2, y+h-6.2, body, ha="center", va="top", fontsize=7, color="#333333", linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
box(1, 22, 19, 19, "1. Knowledge graph", "sourced dam, breach &\nroughness parameters\n(value, range, source)\n+ Copernicus DEM", "#777777")
box(25, 22, 22, 19, "2a. FV-SWE (2DH)", "HLL + hydrostatic\nreconstruction, Manning\n(same equation class\nas Delft3D-FLOW/FM)", C1)
box(25, 1, 22, 18, "2b. WCSPH", "near-field breach /\nnon-hydrostatic surge\nWendland kernel, δ-SPH", C2)
box(52, 22, 22, 19, "3. Scenario ensemble", "ranges queried from\nthe graph (SPARQL),\nLatin hypercube\nh$_{max}$, arrival, area", C1)
box(52, 1, 22, 18, "4. ML surrogate", "POD of h$_{max}$ fields\n+ GPR / NN regression\nhybrid breach-model\nfeatures", C3)
box(79, 22, 20, 19, "5. Outputs (GUI)", "inundation extent\narrival-time maps\n.shp / .kml export", "#777777")
box(79, 1, 20, 18, "6. Near-real-time", "GEE Sentinel-1 SAR\nflood extent for\nmodel checking", "#777777")
arrow(20.5, 31.5, 24.5, 31.5); arrow(47.5, 31.5, 51.5, 31.5); arrow(63, 21.5, 63, 19.5)
arrow(74.5, 10, 78.5, 28); arrow(74.5, 31.5, 78.5, 31.5); arrow(36, 21.5, 36, 19.5)
arrow(89, 19.5, 89, 21.5)
ax.text(37.5, 20.5, "compare", fontsize=6.5, color="#555555", va="center")
ax.text(64, 20.5, "train", fontsize=6.5, color="#555555", va="center")
ax.text(90, 20.5, "validate", fontsize=6.5, color="#555555", va="center")
fig.tight_layout(pad=0.2); fig.savefig("fig1_framework.png", dpi=300)
