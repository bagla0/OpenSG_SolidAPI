"""Cross-section figures for st15 / st12: thin-walled (1D shell line) vs solid
(FEniCSx 2D mesh), each annotated with the material frame e1 (beam axis, out of
page), e2 (tangent), e3 (through-thickness normal, inward).  Frames are read from
the YAML elementOrientations (9 numbers/elem = [e1(3), e2(3), e3(3)])."""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
ST = {
    "st15": dict(shell=r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml",
                 solid=os.path.join(FE, "2Dsolid_VABS_15.yaml")),
    "st12": dict(shell=os.path.join(FE, "1Dshell_12.yaml"),
                 solid=os.path.join(FE, "2Dsolid_12.yaml")),
}
C_E1, C_E2, C_E3 = "tab:green", "tab:blue", "tab:red"


def _row(r):
    if isinstance(r, str):
        return r.strip("[]").split()
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        return r[0].strip("[]").split()
    return [str(v) for v in r]


def load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(v) for v in _row(n)] for n in d["nodes"]])
    elems = [[int(v) for v in _row(e)] for e in d["elements"]]
    ori = np.array([[float(v) for v in _row(o)] for o in d["elementOrientations"]]) \
        if "elementOrientations" in d else None
    return nodes, elems, ori


def _frame_legend(ax, span):
    """A reference triad in a corner: e1 out-of-page, e2, e3."""
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], color=C_E1, marker="o", mfc="none", ls="none", ms=10, label="e1  beam axis (out of page)"),
         ml.Line2D([], [], color=C_E2, lw=2.5, label="e2  tangent (along wall)"),
         ml.Line2D([], [], color=C_E3, lw=2.5, label="e3  normal (through-thickness, inward)")]
    ax.legend(handles=h, loc="upper right", fontsize=9, framealpha=0.9)


def plot_station(name, cfg, out):
    fig, (axT, axS) = plt.subplots(2, 1, figsize=(13, 8.5))

    # ---- thin-walled (1D shell line) ----
    n, el, ori = load(cfg["shell"]); xy = n[:, :2]
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    s = 0.05 * span
    for e in el:
        a, b = e[0] - 1, e[-1] - 1
        axT.plot(xy[[a, b], 0], xy[[a, b], 1], "-", color="0.25", lw=0.9, zorder=2)
    ne = len(el); step = max(1, ne // 16)
    for i in range(0, ne, step):
        e = el[i]; m = 0.5 * (xy[e[0] - 1] + xy[e[-1] - 1])
        e2 = ori[i][3:5]; e3 = ori[i][6:8]
        for vec, col in ((e2, C_E2), (e3, C_E3)):
            v = vec / (np.linalg.norm(vec) + 1e-30) * s
            axT.annotate("", xy=m + v, xytext=m,
                         arrowprops=dict(color=col, lw=1.6, arrowstyle="-|>", shrinkA=0, shrinkB=0), zorder=4)
        axT.plot(*m, "o", color=C_E1, mfc="none", ms=5, zorder=5)        # e1 out-of-page dot
    axT.set_title("%s  THIN-WALLED  (1D shell reference line)" % name, fontsize=12)
    axT.set_aspect("equal"); axT.grid(alpha=0.25); _frame_legend(axT, span)

    # ---- solid (FEniCSx 2D mesh) ----
    ns, els, oris = load(cfg["solid"]); sxy = ns[:, :2]
    for e in els:
        idx = [k - 1 for k in e] + [e[0] - 1]
        axS.plot(sxy[idx, 0], sxy[idx, 1], "-", color="0.6", lw=0.25, zorder=1)
    span2 = max(np.ptp(sxy[:, 0]), np.ptp(sxy[:, 1])); s2 = 0.04 * span2
    nes = len(els); step = max(1, nes // 36)
    for i in range(0, nes, step):
        e = els[i]; cpt = sxy[[k - 1 for k in e]].mean(axis=0)
        e2 = oris[i][3:5]; e3 = oris[i][6:8]
        for vec, col in ((e2, C_E2), (e3, C_E3)):
            v = vec / (np.linalg.norm(vec) + 1e-30) * s2
            axS.annotate("", xy=cpt + v, xytext=cpt,
                         arrowprops=dict(color=col, lw=1.3, arrowstyle="-|>", shrinkA=0, shrinkB=0), zorder=4)
        axS.plot(*cpt, "o", color=C_E1, mfc="none", ms=4, zorder=5)
    axS.set_title("%s  SOLID  (FEniCSx 2D mesh, %d elem)" % (name, len(els)), fontsize=12)
    axS.set_aspect("equal"); axS.grid(alpha=0.25); _frame_legend(axS, span2)

    fig.suptitle("%s cross-section + material frame (e1 beam-axis out of page, e2 tangent, e3 inward normal)"
                 % name, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    for nm, cfg in ST.items():
        plot_station(nm, cfg, os.path.join(HERE, "%s_sections.png" % nm))
