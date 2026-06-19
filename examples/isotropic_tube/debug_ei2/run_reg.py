"""Penalty regression: Timo 6x6 diagonal for the open airfoils (st15, st12) and
the iso tube, at whatever OPENSG_PENALTY is set. Checks that raising the C1
penalty to fix the closed-tube EI2!=EI3 does not disturb the open-section cases."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh

pen = os.environ.get("OPENSG_PENALTY", "default(1e9)")
print("# OPENSG_PENALTY=%s" % pen)
for m in ["1Dshell_15", "1Dshell_12", "1Dtube_iso_0.01", "1Dtube_iso_0.06", "1Dtube_iso_0.2"]:
    sm = ShellBounMesh("data/%s.yaml" % m)
    abd = sm.compute_ABD(); ABD = abd[0] if isinstance(abd, (tuple, list)) else abd
    EB, D = sm.compute_timo(ABD)[0:2]
    d = np.diag(np.asarray(D))
    print("%-16s EA=%.5e GA2=%.5e GA3=%.5e GJ=%.5e EI2=%.5e EI3=%.5e"
          % (m, d[0], d[1], d[2], d[3], d[4], d[5]), flush=True)
