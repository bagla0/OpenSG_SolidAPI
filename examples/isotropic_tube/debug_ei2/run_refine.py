"""Run the FEniCS shell on the refinement tubes (NC=160,320,640, seam at phi=0)."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

print("# NC EI2 EI3 gap%%  (solid EI=2.198475e9)")
for NC in [160, 320, 640]:
    sm = ShellBounMesh("data/1Dref_iso_%d.yaml" % NC)
    abd = sm.compute_ABD(); ABD = abd[0] if isinstance(abd, (tuple, list)) else abd
    EB, D = sm.compute_timo(ABD)[0:2]
    d = np.diag(np.asarray(D))
    gap = 100.0 * (d[4] - d[5]) / (0.5 * (d[4] + d[5]))
    e2 = 100.0 * (d[4] - 2.198475e9) / 2.198475e9
    e3 = 100.0 * (d[5] - 2.198475e9) / 2.198475e9
    print("REF %d EI2=%.5e EI3=%.5e gap=%+.2f%%  (EI2 %+.2f%%, EI3 %+.2f%% vs solid)"
          % (NC, d[4], d[5], gap, e2, e3), flush=True)
