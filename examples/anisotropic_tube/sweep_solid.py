"""FEniCS 2D-solid full Timo 6x6 for the [45/-45] annulus (+45 outer / -45 inner,
verified), swept over h/R. Ground truth. Writes results/solid.txt (full 36-value 6x6)."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
MD = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/aniso_tube/solid_meshes"
RES = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/aniso_tube/results"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.makedirs(RES, exist_ok=True)
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]
lines = ["# tag hR <36 values of 6x6 row-major>"]
for hR in hRs:
    sm = SolidBounMesh(MD + "/tube_aniso_hR%s.yaml" % hR)
    mp, _ = sm.material_database
    C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
    s = "SOLID %s " % hR + " ".join("%.6e" % v for v in C6.flatten())
    print("SOLID", hR, "ok", flush=True); lines.append(s)
open(os.path.join(RES, "solid.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/solid.txt")
