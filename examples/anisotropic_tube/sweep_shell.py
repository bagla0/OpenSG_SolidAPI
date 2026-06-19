"""FEniCS Kirchhoff (DG-penalty) shell full Timo 6x6 for the [45/-45] tube, swept
over h/R at center/OML/IML. Writes results/shell.txt with the full 36-value 6x6."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
RES = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/aniso_tube/results"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.makedirs(RES, exist_ok=True)
from opensg.mesh.segment import ShellBounMesh

hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]
lines = ["# tag hR <36 values of 6x6 row-major>"]
for tag, pat, frac in [("SHELL_C", "data/1Dtube_aniso_%s.yaml", 0.5),
                       ("SHELL_O", "data/1Dtube_aniso_OML_%s.yaml", 0.0)]:   # center + OML only
    for hR in hRs:
        sm = ShellBounMesh(pat % hR)
        ABD = sm.compute_ABD(frac=frac)[0]
        EB, D = sm.compute_timo(ABD)[0:2]
        D = np.asarray(D)
        s = "%s %s " % (tag, hR) + " ".join("%.6e" % v for v in D.flatten())
        print(tag, hR, "ok", flush=True); lines.append(s)
open(os.path.join(RES, "shell.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/shell.txt")
