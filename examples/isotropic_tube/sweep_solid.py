"""FEniCS 2D-solid Timoshenko 6x6 for the iso-cylinder annulus, swept over h/R.
This is the ground-truth reference. Writes results/solid.txt: SOLID hR EA..EI3."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
MD = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/benchmark_tube/solid_meshes/meshes"
RES = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/cylinder_study/results"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.makedirs(RES, exist_ok=True)
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]
lines = ["# tag hR EA GA2 GA3 GJ EI2 EI3"]
for hR in hRs:
    sm = SolidBounMesh(MD + "/tube_iso_hR%s.yaml" % hR)
    mp, _ = sm.material_database
    C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
    d = np.diag(C6)
    s = "SOLID %s %.6e %.6e %.6e %.6e %.6e %.6e" % (hR, d[0], d[1], d[2], d[3], d[4], d[5])
    print(s, flush=True); lines.append(s)
open(os.path.join(RES, "solid.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/solid.txt")
