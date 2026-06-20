"""Why is center NOT uniformly the best reference for st15 (unlike the uniform-wall
tube)?  (1) Plot the OML/center/IML geometry -- the frac offset moves each node inward
by its LOCAL thickness, so the thick spar caps distort while thin skins barely move.
(2) List the per-term % error at each reference AND the term's size relative to EA, to
separate genuine reference sensitivity from noise-floor (tiny-coupling) % blow-up."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\opensg_jax")
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from st_oml_iml_table import shell_6x6, read_solid, ST, LBL

HERE = os.path.dirname(__file__)
cfg = ST["st15"]

# ---- (1) geometry distortion ----
n3d, elements, mat_db, layup_db, e2l = load_yaml(cfg["yaml"])
nodes, cells, lpe = read_mesh(n3d, elements, e2l)
e3 = element_e3_from_yaml(cfg["yaml"])
tloc = np.array([sum(layup_db[lpe[i]]["thick"]) for i in range(len(cells))])
print("st15 wall thickness: min=%.4f  max=%.4f  (max/min=%.1fx) -> non-uniform" % (
    tloc.min(), tloc.max(), tloc.max() / tloc.min()))

fig, ax = plt.subplots(figsize=(14, 5))
for frac, col, lab in [(0.0, "k", "OML (frac 0)"), (0.5, "tab:blue", "center (0.5)"), (1.0, "tab:red", "IML (1.0)")]:
    nd = nodes[:, :2] if frac == 0 else offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)[:, :2]
    for c in cells:
        a, b = c[0], c[-1]
        ax.plot(nd[[a, b], 0], nd[[a, b], 1], "-", color=col, lw=0.9)
    ax.plot([], [], "-", color=col, lw=2, label=lab)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=10)
ax.set_title("st15: reference geometry at OML / center / IML\n"
             "frac offset = inward by LOCAL thickness -> thick spar caps pull in, skins barely move (non-uniform distortion)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "st15_reference_geometry.png"), dpi=130)
print("wrote st15_reference_geometry.png")

# ---- (2) term-by-term: % error AND size vs EA ----
S = read_solid(cfg)
res = {f: shell_6x6(cfg, f) for f in (0.0, 0.5, 1.0)}
EA = abs(S[0, 0])
print("\n%-11s %10s %7s | %6s %6s %6s   (K = Kirchhoff)" % ("term", "solid", "vs EA", "K-OML", "K-cen", "K-IML"))
for a in range(6):
    for b in range(a, 6):
        v = S[a, b]
        if abs(v) < 1e5:
            continue
        rel = abs(v) / EA * 100
        kf = [100 * (res[f][1][a, b] - v) / v for f in (0.0, 0.5, 1.0)]
        flag = "  <- NOISE (tiny term)" if rel < 0.5 else ""
        print("%-2d,%-2d %-5s %10.2e %6.2f%% | %+6.1f %+6.1f %+6.1f%s" % (
            a + 1, b + 1, "%s-%s" % (LBL[a][:3], LBL[b][:3]), v, rel, kf[0], kf[1], kf[2], flag))
