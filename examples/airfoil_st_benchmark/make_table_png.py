"""Render the st15 / st12 Timoshenko 6x6 benchmark tables (% error vs 2D solid, at
OML / center / IML references, JAX Kirchhoff + RM) as colour-coded PNG images."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from st_oml_iml_table import shell_6x6, read_solid, ST, LBL

HERE = os.path.dirname(__file__)


def cell_color(e):
    a = abs(e)
    return "#c8e6c9" if a < 5 else "#fff9c4" if a < 15 else "#ffcdd2"   # green / yellow / red


def make(name, cfg, out):
    S = read_solid(cfg)
    res = {f: shell_6x6(cfg, f) for f in (0.0, 0.5, 1.0)}
    rows, colors = [], []
    for a in range(6):
        for b in range(a, 6):
            v = S[a, b]
            if abs(v) < 1e6:                              # skip noise-floor terms
                continue
            nm = "%d,%d %s-%s" % (a + 1, b + 1, LBL[a][:3], LBL[b][:3])
            kf = [100 * (res[f][1][a, b] - v) / v for f in (0.0, 0.5, 1.0)]
            rm = [100 * (res[f][0][a, b] - v) / v for f in (0.0, 0.5, 1.0)]
            rows.append([nm, "%.3e" % v] + ["%+.1f" % x for x in kf] + ["%+.1f" % x for x in rm])
            colors.append(["white", "#ececec"] + [cell_color(x) for x in kf] + [cell_color(x) for x in rm])
    cols = ["term", "solid", "K-OML", "K-cen", "K-IML", "RM-OML", "RM-cen", "RM-IML"]
    fig, ax = plt.subplots(figsize=(11.5, 0.46 * len(rows) + 1.4))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, cellColours=colors,
                 colColours=["#cfd8dc"] * len(cols), loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title("%s   Timoshenko 6x6 stiffness:  %% error vs 2D solid   (OML / center / IML reference)\n"
                 "cell shading: green |err|<5%%,  yellow <15%%,  red >15%%   (K = JAX Kirchhoff, RM = JAX Reissner-Mindlin)"
                 % name, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    for nm, cfg in ST.items():
        make(nm, cfg, os.path.join(HERE, "%s_table.png" % nm))
