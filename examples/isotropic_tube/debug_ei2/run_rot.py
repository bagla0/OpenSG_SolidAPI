"""Run the FEniCS DG-penalty shell on the seam-rotated tubes (phi=0,45,90) and
report EI2, EI3 and the gap. Decisive: seam-localized C1 defect -> EI2/EI3 SWAP
at 90 and EQUALIZE at 45; coordinate/frame defect -> no change."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

print("# phi_deg EA GA2 GA3 GJ EI2 EI3   (EI2-EI3)/avg %%")
for phi in [0, 45, 90]:
    sm = ShellBounMesh("data/1Drot_iso_%d.yaml" % phi)
    abd = sm.compute_ABD(); ABD = abd[0] if isinstance(abd, (tuple, list)) else abd
    EB, D = sm.compute_timo(ABD)[0:2]
    d = np.diag(np.asarray(D))
    gap = 100.0 * (d[4] - d[5]) / (0.5 * (d[4] + d[5]))
    print("ROT %d %.5e %.5e %.5e %.5e %.5e %.5e  gap=%+.2f%%"
          % (phi, d[0], d[1], d[2], d[3], d[4], d[5], gap), flush=True)
