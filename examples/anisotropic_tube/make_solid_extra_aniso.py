"""Generate the two thick-wall [45/-45] annulus solid meshes (h/R = 0.35, 0.5)
that extend the existing tube_aniso_hR{0.01..0.2} set, reusing gen_tube + ANI_LAYUP
from the benchmark solid-mesh generator. R_mid=1.0, ANI ply, [-45 inner, 45 outer]."""
import sys, os
SM = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\solid_meshes"
sys.path.insert(0, SM)
from generate_solid_meshes import gen_tube, ANI, ANI_LAYUP, OUT

R = 1.0
for hr in [0.35, 0.5]:
    p = os.path.join(OUT, "tube_aniso_hR%s.yaml" % hr)
    gen_tube(p, R, hr * R, ANI, ANI_LAYUP)
    print("wrote", os.path.basename(p), " ri=%.3f ro=%.3f [-45/45]" % (R - hr * R / 2, R + hr * R / 2))
