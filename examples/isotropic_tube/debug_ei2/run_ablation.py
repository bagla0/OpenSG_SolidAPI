"""EI2 vs EI3 for the iso tube (center ref) with the C1-penalty consistency-term
ablation toggled by env OPENSG_PURE_PENALTY. Decisive test of rank-1 hypothesis."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

SOL = {"0.01": 2.198475e9, "0.06": 1.320239e10, "0.2": 4.440817e10}
mode = os.environ.get("OPENSG_PURE_PENALTY", "0")
print("# OPENSG_PURE_PENALTY=%s  (0=baseline Nitsche, 1=ablated pure penalty)" % mode)
print("# hR EA GJ EI2 EI3 gap%%  EI2vs_solid%% EI3vs_solid%%")
for hR in ["0.01", "0.06", "0.2"]:
    sm = ShellBounMesh("data/1Dtube_iso_%s.yaml" % hR)
    abd = sm.compute_ABD(); ABD = abd[0] if isinstance(abd, (tuple, list)) else abd
    EB, D = sm.compute_timo(ABD)[0:2]
    d = np.diag(np.asarray(D))
    gap = 100.0 * (d[4] - d[5]) / (0.5 * (d[4] + d[5]))
    e2 = 100.0 * (d[4] - SOL[hR]) / SOL[hR]
    e3 = 100.0 * (d[5] - SOL[hR]) / SOL[hR]
    print("%s EA=%.4e GJ=%.4e EI2=%.5e EI3=%.5e gap=%+.2f%%  (EI2 %+.2f%%, EI3 %+.2f%%)"
          % (hR, d[0], d[3], d[4], d[5], gap, e2, e3), flush=True)
