"""Diagnose the OML reference error for the [45/-45] tube.  The solid is at the
beam axis (= center).  Geometric line-radius law (iso): EA/EI ~ +eps, GJ/GA ~ -2eps,
eps=(h/R)/2.  If the aniso OML deviates far from this, a B-matrix (outer-ref) bug."""
import sys, os, numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\opensg_jax")
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
from rm_tube import rm_tube_6x6

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data"
SOL = {}
for ln in open(os.path.join(HERE, "results", "solid.txt")):
    if ln.startswith("SOLID"):
        p = ln.split(); SOL[p[1]] = np.array([float(x) for x in p[2:]]).reshape(6, 6)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
for hR in ["0.01", "0.03", "0.06", "0.12"]:
    y = os.path.join(DATA, "shell_aniso_%s.yaml" % hR)
    S = SOL[hR]; d_s = np.diag(S)
    eps = float(hR) / 2.0
    _, Kc, _ = timoshenko_from_yaml(y, frac=0.5); Kc = np.asarray(Kc)
    _, Ko, _ = timoshenko_from_yaml(y, frac=0.0); Ko = np.asarray(Ko)
    print("h/R=%s  eps=(h/R)/2=%.3f  geometric law: EA/EI ~ +%.0f%%, GJ/GA ~ -%.0f%%" % (
        hR, eps, 100 * eps, 200 * eps))
    print("  term  center%%err  OML%%err   (center should be ~0)")
    for i, lb in enumerate(LBL):
        ce = 100 * (np.diag(Kc)[i] - d_s[i]) / d_s[i]
        oe = 100 * (np.diag(Ko)[i] - d_s[i]) / d_s[i]
        print("  %-4s  %+8.1f   %+8.1f" % (lb, ce, oe))
    # B-matrix tell: ext-twist (0,3) and shear-bend (1,4) couplings at OML vs center
    print("  ext-twist: center %+.3e  OML %+.3e  (solid %+.3e)" % (Kc[0, 3], Ko[0, 3], S[0, 3]))
    print("  shear-bend: center %+.3e  OML %+.3e  (solid %+.3e)" % (Kc[1, 4], Ko[1, 4], S[1, 4]))
    print()
