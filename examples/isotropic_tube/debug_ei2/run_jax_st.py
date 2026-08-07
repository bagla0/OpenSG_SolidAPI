"""JAX Hermite-C1 shell Timo 6x6 reference for st15/st12 (airfoil, OML frac=0)."""
import sys, os
import numpy as np
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
DATA = os.path.join(HERE, "training data", "opensg-FEniCS", "data")
for m in ["1Dshell_15", "1Dshell_12"]:
    _, K, _ = timoshenko_from_yaml(os.path.join(DATA, m + ".yaml"), frac=0.0)
    d = np.diag(np.asarray(K))
    print("%s JAX: EA=%.5e GA2=%.5e GA3=%.5e GJ=%.5e EI2=%.5e EI3=%.5e"
          % (m, d[0], d[1], d[2], d[3], d[4], d[5]))
