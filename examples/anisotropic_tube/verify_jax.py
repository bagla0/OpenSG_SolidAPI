"""Dump the full JAX Kirchhoff Timoshenko 6x6 for the [45/-45] tube (center, frac=0.5)
so the coupling terms can be checked against the FEniCS solid (same physical layup?)."""
import sys, os
import numpy as np
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
sys.path.insert(0, os.path.join(HERE, "cylinder_study"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
from rm_tube import rm_tube_6x6
np.set_printoptions(precision=3, suppress=False, linewidth=160)
y = os.path.join(HERE, "benchmark_tube", "data", "shell_aniso_0.06.yaml")
_, K, _ = timoshenko_from_yaml(y, frac=0.5)
K = np.asarray(K)
print("# JAX Kirchhoff [45/-45] tube h/R=0.06, center (frac=0.5), full 6x6")
print("# order [ext, shear2, shear3, twist, bend2, bend3]")
for r in range(6):
    print(" ".join("%12.4e" % K[r, c] for c in range(6)))
RM, _ = rm_tube_6x6(y, frac=0.5)
RM = np.asarray(RM)
print("# JAX RM [45/-45] tube h/R=0.06, center (frac=0.5), full 6x6")
for r in range(6):
    print(" ".join("%12.4e" % RM[r, c] for c in range(6)))
