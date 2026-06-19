"""Dump the full FEniCS 2D-solid Timoshenko 6x6 for the [45/-45] annulus (h/R=0.06)
to compare coupling-term signs against the JAX shell (same physical layup?)."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
MD = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/benchmark_tube/solid_meshes/meshes"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
np.set_printoptions(precision=3, linewidth=160)
sm = SolidBounMesh(MD + "/tube_aniso_hR0.06.yaml")
mp, _ = sm.material_database
C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
print("# FEniCS solid [-45/45] annulus h/R=0.06, full 6x6")
print("# order [ext, shear2, shear3, twist, bend2, bend3]")
for r in range(6):
    print(" ".join("%12.4e" % C6[r, c] for c in range(6)), flush=True)
