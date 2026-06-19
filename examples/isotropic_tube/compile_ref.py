"""Reference comparison for the isotropic cylinder: IML vs center vs OML, plotted as
% error vs the FEniCS 2D solid (VABS) for all six Timoshenko terms, for both the
FEniCS Kirchhoff (DG penalty) shell and the JAX Kirchhoff (Hermite C1) shell.

Shows how the choice of where the shell line sits (inner / mid / outer surface)
brackets the true (solid) value: center ~ 0, OML over-counts EA/EI (under GJ/GA),
IML the mirror. Reads results/{shell,jax,solid}.txt. Uses h/R only."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\cylinder_study"
RES = os.path.join(CS, "results")
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35"]  # h/R=0.5 dropped (off-scale, far past thick-shell limit)
hR = np.array([float(h) for h in HR])
NAMES = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def load(fn):
    D = {}
    for ln in open(os.path.join(RES, fn)):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        D.setdefault(p[0], {})[p[1]] = [float(x) for x in p[2:8]]
    return D


data = {}
for fn in ("jax.txt", "solid.txt", "shell.txt"):
    data.update(load(fn))
diag = lambda tag, j: np.array([data[tag][h][j] for h in HR])
SOLID = lambda j: diag("SOLID", j)

# reference -> colour (line position: inner / mid / outer surface)
REFS = [("I", "IML  (inner, R-t/2)", "tab:red"),
        ("C", "center (mid, R)", "tab:green"),
        ("O", "OML  (outer, R+t/2)", "tab:blue")]
# method -> (tag prefix, label, linestyle, marker)
METHODS = [("SHELL", "FEniCS Kirchhoff (DG)", "-", "o"),
           ("KIRCH", "JAX Kirchhoff (Hermite)", "--", "s"),
           ("RM", "JAX RM (Hermite)", ":", "^")]

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for j, nm in enumerate(NAMES):
    a = ax[j // 3, j % 3]
    s = SOLID(j)
    a.axvspan(0.1, 0.35, color="gray", alpha=0.07, zorder=0)
    a.axvline(0.1, color="gray", ls="-.", lw=1.2, zorder=1, label="thin | thick shell (h/R=0.1)")
    a.axhline(0, color="k", lw=1.4, label="FEniCS Solid (VABS)")
    for mtag, mlab, ls, mk in METHODS:
        for rsuf, rlab, col in REFS:
            err = 100.0 * (diag(mtag + "_" + rsuf, j) - s) / s
            a.plot(hR, err, color=col, ls=ls, lw=1.9, marker=mk, ms=6,
                   mfc=("none" if ls != "-" else col), label="%s - %s" % (mlab, rlab))
    a.set_xlabel("h/R"); a.set_ylabel("% error vs solid")
    a.set_title("%s   (%% error vs FEniCS solid)" % nm)
    a.grid(alpha=0.3)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=True)
fig.suptitle("Isotropic cylinder R=1: reference comparison IML vs center vs OML (% error vs FEniCS solid)", fontsize=13)
fig.tight_layout(rect=[0, 0.08, 1, 0.97])
out = os.path.join(CS, "cylinder_refcompare.png")
fig.savefig(out, dpi=120)
print("wrote", out)

# table: FEniCS shell error at each reference
print("\nFEniCS Kirchhoff shell %% error vs solid, per reference:")
for rsuf, rlab, _ in REFS:
    print("# " + rlab)
    print("  %-5s " % "" + "  ".join("h/R=%-5s" % h for h in HR))
    for j, nm in enumerate(NAMES):
        err = 100.0 * (diag("SHELL_" + rsuf, j) - SOLID(j)) / SOLID(j)
        print("  %-4s " % nm + "  ".join("%+7.2f" % e for e in err))
