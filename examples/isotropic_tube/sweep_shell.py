"""FEniCS Kirchhoff (DG-penalty) thin-walled shell Timoshenko 6x6 for the iso
cylinder, swept over h/R, at CENTER (1Dtube_iso_<hR>) and OML (1Dtube_iso_OML_<hR>).
Writes results/shell.txt: rows  SHELL_C / SHELL_O  hR  EA GA2 GA3 GJ EI2 EI3."""
import sys, os
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
RES = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/cylinder_study/results"
sys.path.insert(0, PKG)
os.chdir(PKG)
os.makedirs(RES, exist_ok=True)
from opensg.mesh.segment import ShellBounMesh

hRs = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]
lines = ["# tag hR EA GA2 GA3 GJ EI2 EI3"]
# center: line on the wall mid-surface (R_mid) -> ABD frac=0.5 (mid-ply reference);
# OML: line on the outer surface (R_oml) -> ABD frac=0 (OML reference).
for tag, pat, frac in [("SHELL_C", "data/1Dtube_iso_%s.yaml", 0.5),
                       ("SHELL_O", "data/1Dtube_iso_OML_%s.yaml", 0.0)]:
    for hR in hRs:
        sm = ShellBounMesh(pat % hR)
        abd = sm.compute_ABD(frac=frac); ABD = abd[0] if isinstance(abd, (tuple, list)) else abd
        EB, D = sm.compute_timo(ABD)[0:2]
        d = np.diag(np.asarray(D))
        s = "%s %s %.6e %.6e %.6e %.6e %.6e %.6e" % (tag, hR, d[0], d[1], d[2], d[3], d[4], d[5])
        print(s, flush=True); lines.append(s)
open(os.path.join(RES, "shell.txt"), "w").write("\n".join(lines) + "\n")
print("wrote results/shell.txt")
