"""[45/-45] tube thickness sweep: % error vs FEniCS solid (VABS) for ALL non-zero
Timoshenko 6x6 terms -- 6 diagonal (EA,GA2,GA3,GJ,EI2,EI3) + 3 coupling
(ext-twist, shear2-bend2, shear3-bend3) -- for JAX Kirchhoff, JAX RM, and the
FEniCS Kirchhoff shell, at center and OML references. NO analytical (none in
closed form for [45/-45]). Reads results/{jax,solid,shell}.txt (full 6x6)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\aniso_tube"
RES = os.path.join(CS, "results")
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]
hR = np.array([float(h) for h in HR])

# non-zero terms (label, i, j) 0-indexed; symmetric matrix
TERMS = [("EA", 0, 0), ("GA2", 1, 1), ("GA3", 2, 2), ("GJ", 3, 3), ("EI2", 4, 4), ("EI3", 5, 5),
         ("ext-twist (1,4)", 0, 3), ("shear2-bend2 (2,5)", 1, 4), ("shear3-bend3 (3,6)", 2, 5)]


def load(fn):
    D = {}
    for ln in open(os.path.join(RES, fn)):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        M = np.array([float(x) for x in p[2:38]]).reshape(6, 6)
        D.setdefault(p[0], {})[p[1]] = 0.5 * (M + M.T)        # symmetrize
    return D


data = {}
for fn in ("jax.txt", "solid.txt", "shell.txt"):
    data.update(load(fn))
term = lambda tag, i, j: np.array([data[tag][h][i, j] for h in HR])
SOL = lambda i, j: term("SOLID", i, j)

CURVES = [
    ("KIRCH_C", "JAX Kirchhoff - center", "tab:blue", "-", "o"),
    ("KIRCH_O", "JAX Kirchhoff - OML", "tab:blue", "--", "o"),
    ("RM_C", "JAX RM - center", "tab:orange", "-", "s"),
    ("RM_O", "JAX RM - OML", "tab:orange", "--", "s"),
    ("SHELL_C", "FEniCS Kirchhoff (DG) - center", "tab:red", "-", "v"),
    ("SHELL_O", "FEniCS Kirchhoff (DG) - OML", "tab:red", "--", "v"),
]

fig, ax = plt.subplots(3, 3, figsize=(16, 13))
for k, (nm, i, j) in enumerate(TERMS):
    a = ax[k // 3, k % 3]
    s = SOL(i, j)
    a.axvspan(0.1, 0.4, color="gray", alpha=0.07, zorder=0)
    a.axvline(0.1, color="gray", ls="-.", lw=1.2, zorder=1, label="thin | thick shell (h/R=0.1)")
    a.axhline(0, color="k", lw=1.4, label="FEniCS Solid (VABS)")
    for tag, lab, col, ls, mk in CURVES:
        err = 100.0 * (term(tag, i, j) - s) / s
        a.plot(hR, err, color=col, ls=ls, lw=1.9, marker=mk, ms=6,
               mfc=("none" if ls == "--" else col), label=lab, zorder=(4 if "RM" in tag else 3))
    a.set_xlabel("h/R"); a.set_ylabel("% error vs solid")
    kind = "diagonal" if i == j else "coupling"
    a.set_title("%s  (%s)" % (nm, kind))
    a.grid(alpha=0.3)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True)
fig.suptitle("[45/-45] composite tube R=1: Timoshenko 6x6 % error vs FEniCS solid (VABS), all non-zero terms", fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
out = os.path.join(CS, "aniso_pcterr.png")
fig.savefig(out, dpi=115)
print("wrote", out)

# text table (center)
print("\n%% error vs solid (center reference):")
for tag, lab, *_ in CURVES:
    if not tag.endswith("_C"):
        continue
    print("# " + lab)
    for nm, i, j in TERMS:
        err = 100.0 * (term(tag, i, j) - SOL(i, j)) / SOL(i, j)
        print("  %-20s " % nm + "  ".join("%+7.1f" % e for e in err))
