"""Debug the FEniCS [45/-45] spikes: run h/R=0.12 (worst) center, dump the 2<->3
symmetric terms (EI2/EI3, GA2/GA3, coupling) at the active penalty (set via
OPENSG_PENALTY). Clean = EI2==EI3, GA2==GA3, shear2-bend2 == shear3-bend3."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
os.chdir(PKG)
from opensg.mesh.segment import ShellBounMesh
sm = ShellBounMesh("data/1Dtube_aniso_0.12.yaml")
ABD = sm.compute_ABD(frac=0.5)[0]
EB, D = sm.compute_timo(ABD)[0:2]
D = 0.5 * (np.asarray(D) + np.asarray(D).T)
pen = os.environ.get("OPENSG_PENALTY", "default(2e2*max_E=7.4e12)")
print("penalty=%s  h/R=0.12 [45/-45] center" % pen)
print("  EA=%.4e GJ=%.4e" % (D[0, 0], D[3, 3]))
print("  EI2=%.4e EI3=%.4e  (gap %+.1f%%)" % (D[4, 4], D[5, 5], 100 * (D[4, 4] - D[5, 5]) / D[5, 5]))
print("  GA2=%.4e GA3=%.4e  (gap %+.1f%%)" % (D[1, 1], D[2, 2], 100 * (D[1, 1] - D[2, 2]) / D[2, 2]))
print("  ext-tw(1,4)=%.4e" % D[0, 3])
print("  sh2-b2(2,5)=%.4e  sh3-b3(3,6)=%.4e  (gap %+.1f%%)" % (D[1, 4], D[2, 5], 100 * (D[1, 4] - D[2, 5]) / D[2, 5]), flush=True)
