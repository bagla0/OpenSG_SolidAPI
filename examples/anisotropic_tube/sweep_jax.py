"""JAX Kirchhoff (Hermite-C1) + JAX RM full Timoshenko 6x6 for the [45/-45] tube,
swept over h/R, at center(frac=0.5)/OML(0.0)/IML(1.0). Writes results/jax.txt with
the FULL 36-value 6x6 per row (TAG hR v00 v01 ... v55) so coupling terms are kept."""
import sys, os
import numpy as np
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "aniso_tube"))
sys.path.insert(0, os.path.join(HERE, "cylinder_study"))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
from rm_tube import rm_tube_6x6

DATA = os.path.join(HERE, "benchmark_tube", "data")
RES = os.path.join(HERE, "aniso_tube", "results")
os.makedirs(RES, exist_ok=True)
hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]
lines = ["# tag hR <36 values of 6x6 row-major>"]


def row(tag, hR, M):
    M = np.asarray(M)
    s = "%s %s " % (tag, hR) + " ".join("%.6e" % v for v in M.flatten())
    print(tag, hR, "ok", flush=True); lines.append(s)


for hR in hRs:
    y = os.path.join(DATA, "shell_aniso_%s.yaml" % hR)
    for frac, suf in [(0.5, "C"), (0.0, "O")]:   # center + OML only
        _, K, _ = timoshenko_from_yaml(y, frac=frac)
        row("KIRCH_" + suf, hR, K)
        RM, _ = rm_tube_6x6(y, frac=frac)
        row("RM_" + suf, hR, RM)
open(os.path.join(RES, "jax.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/jax.txt")
