"""Isotropic-cylinder thickness sweep (h/R 0.01 -> 0.5). Plots % error of every
shell method vs the FEniCS 2D-solid (ground truth) for all SIX Timoshenko
diagonal stiffnesses [EA, GA2, GA3, GJ, EI2, EI3], comparing CENTER (mid-wall)
and OML (outer-surface) references.

Methods:  JAX Kirchhoff (Hermite C1) | JAX RM | FEniCS Kirchhoff (DG penalty).
Reads results/{jax,solid,shell}.txt produced by the sweep_*.py drivers."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\cylinder_study"
RES = os.path.join(CS, "results")
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]
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


def diag(tag, j):
    return np.array([data[tag][h][j] for h in HR])


SOLID = lambda j: diag("SOLID", j)

# MSG thin-walled-shell closed-form (closed circular tube, mid-surface reference R=R_mid):
#   EA = E 2piR t ; GA2=GA3 = (1/2) G 2piR t ; GJ = G 2piR^3 t (Bredt) ; EI2=EI3 = E piR^3 t.
# Leading order in t/R (drops the O((t/R)^2) terms the solid annulus keeps), so it is the
# "analytical shell" baseline, NOT the exact ring. Mid-surface => no center/OML choice.
E, nu = 70e9, 0.3
G = E / (2 * (1 + nu))
R = 1.0
tarr = hR * R


def analytical(j):
    EA = E * 2 * np.pi * R * tarr
    GA = 0.5 * G * 2 * np.pi * R * tarr
    GJ = G * 2 * np.pi * R ** 3 * tarr
    EI = E * np.pi * R ** 3 * tarr
    return [EA, GA, GA, GJ, EI, EI][j]


# (tag, label, color, linestyle, marker)  -- RM = orange (high contrast vs Kirchhoff blue)
CURVES = [
    ("KIRCH_C", "JAX Kirchhoff (Hermite C1) - center", "tab:blue", "-", "o"),
    ("KIRCH_O", "JAX Kirchhoff (Hermite C1) - OML", "tab:blue", "--", "o"),
    ("RM_C", "JAX RM (Hermite C1) - center", "tab:orange", "-", "s"),
    ("RM_O", "JAX RM (Hermite C1) - OML", "tab:orange", "--", "s"),
    ("SHELL_C", "FEniCS Kirchhoff (DG penalty) - center", "tab:red", "-", "v"),
    ("SHELL_O", "FEniCS Kirchhoff (DG penalty) - OML", "tab:red", "--", "v"),
]

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for j, nm in enumerate(NAMES):
    a = ax[j // 3, j % 3]
    s = SOLID(j)
    a.axhline(0, color="k", lw=1.4, label="FEniCS Solid (VABS)")
    a.plot(hR, 100.0 * (analytical(j) - s) / s, color="k", ls=":", lw=2.0, marker="D",
           ms=6, mfc="none", label="Analytical (MSG-Thin Walled)", zorder=5)
    for tag, lab, col, ls, mk in CURVES:
        err = 100.0 * (diag(tag, j) - s) / s
        a.plot(hR, err, color=col, ls=ls, lw=2.0, marker=mk, ms=7,
               mfc=("none" if ls == "--" else col), label=lab, zorder=(4 if "RM" in tag else 3))
    a.set_xlabel("h/R"); a.set_ylabel("% error vs solid")
    a.set_title("%s   (%% error vs FEniCS solid)" % nm)
    a.grid(alpha=0.3)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True)
fig.suptitle("Isotropic cylinder R=1: Timoshenko 6x6 % error vs FEniCS solid  (center vs OML reference, h/R to 0.5)",
             fontsize=13)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
out = os.path.join(CS, "cylinder_pcterr.png")
fig.savefig(out, dpi=120)
print("wrote", out)

# ---- text table ----
print("\n%% error vs FEniCS solid (center reference):")
print("%-5s " % "" + "  ".join("h/R=%-5s" % h for h in HR))
print("# MSG shell analytical (thin-wall, mid-surface)")
for j, nm in enumerate(NAMES):
    err = 100.0 * (analytical(j) - SOLID(j)) / SOLID(j)
    print("  %-4s " % nm + "  ".join("%+7.2f" % e for e in err))
for tag, lab, *_ in CURVES:
    if not tag.endswith("_C"):
        continue
    print("# " + lab)
    for j, nm in enumerate(NAMES):
        err = 100.0 * (diag(tag, j) - SOLID(j)) / SOLID(j)
        print("  %-4s " % nm + "  ".join("%+7.2f" % e for e in err))
print("\n%% error vs FEniCS solid (OML reference):")
for tag, lab, *_ in CURVES:
    if not tag.endswith("_O"):
        continue
    print("# " + lab)
    for j, nm in enumerate(NAMES):
        err = 100.0 * (diag(tag, j) - SOLID(j)) / SOLID(j)
        print("  %-4s " % nm + "  ".join("%+7.2f" % e for e in err))
