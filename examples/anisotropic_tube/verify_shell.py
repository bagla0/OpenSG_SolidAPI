"""Dump the full FEniCS Kirchhoff shell Timo 6x6 for the [45/-45] tube (center, frac=0.5)."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh
np.set_printoptions(precision=3, linewidth=160)
sm = ShellBounMesh("data/1Dtube_aniso_0.06.yaml")
ABD = sm.compute_ABD(frac=0.5)[0]
EB, D = sm.compute_timo(ABD)[0:2]
D = np.asarray(D)
print("# FEniCS Kirchhoff shell [45/-45] tube h/R=0.06, center (frac=0.5), full 6x6")
for r in range(6):
    print(" ".join("%12.4e" % D[r, c] for c in range(6)), flush=True)
