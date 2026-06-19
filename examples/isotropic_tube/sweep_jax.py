"""JAX Kirchhoff (Hermite-C1) + JAX Reissner-Mindlin Timoshenko 6x6 for the
isotropic cylinder, swept over h/R (thin -> thick), at BOTH references:
  frac=0.5 -> CENTER (mid-wall)   frac=0.0 -> OML (outer surface).
Writes results/jax.txt: rows  KIRCH_C / KIRCH_O / RM_C / RM_O  hR  EA GA2 GA3 GJ EI2 EI3."""
import sys, os
import numpy as np
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "cylinder_study"))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
from rm_tube import rm_tube_6x6

DATA = os.path.join(HERE, "benchmark_tube", "data")
RES = os.path.join(HERE, "cylinder_study", "results")
os.makedirs(RES, exist_ok=True)
hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]
lines = ["# tag hR EA GA2 GA3 GJ EI2 EI3"]


def row(tag, hR, d):
    s = "%s %s %.6e %.6e %.6e %.6e %.6e %.6e" % (tag, hR, d[0], d[1], d[2], d[3], d[4], d[5])
    print(s, flush=True); lines.append(s)


for hR in hRs:
    y = os.path.join(DATA, "shell_iso_%s.yaml" % hR)
    for frac, suf in [(0.5, "C"), (0.0, "O")]:
        _, K, _ = timoshenko_from_yaml(y, frac=frac)
        row("KIRCH_" + suf, hR, np.diag(np.asarray(K)))
        RM, _ = rm_tube_6x6(y, frac=frac)
        row("RM_" + suf, hR, np.diag(RM))

open(os.path.join(RES, "jax.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/jax.txt")
