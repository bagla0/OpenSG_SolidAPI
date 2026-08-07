"""Generate the two thick-wall isotropic annulus solid meshes (h/R = 0.35, 0.5)
that extend the existing tube_iso_hR{0.01..0.2} set, reusing gen_tube from the
benchmark solid-mesh generator. R_mid = 1.0, E = 70 GPa, nu = 0.3."""
import sys, os
SM = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\solid_meshes"
sys.path.insert(0, SM)
from generate_solid_meshes import gen_tube, ISO, ISO_LAYUP, OUT

R = 1.0
for hr in [0.35, 0.5]:
    p = os.path.join(OUT, "tube_iso_hR%s.yaml" % hr)
    gen_tube(p, R, hr * R, ISO, ISO_LAYUP)
    print("wrote", os.path.basename(p), " ri=%.3f ro=%.3f" % (R - hr * R / 2, R + hr * R / 2))
