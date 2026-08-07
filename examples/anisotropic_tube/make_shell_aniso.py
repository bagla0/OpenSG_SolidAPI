"""Generate JAX-format [45/-45] composite-tube shell meshes shell_aniso_hR.yaml
(ANI ply E1=37, E2=E3=9 GPa, G=4 GPa, nu=0.3; outer ply +45, inner ply -45 -> matches
the solid tube_aniso convention [-45 inner, 45 outer]). Line at the OML (R_oml=R+t/2),
ply inward, e3 = inward normal. Same node/orientation layout as the iso generator;
reproduces the validated shell_aniso_0.01..0.2 and adds 0.35, 0.5."""
import yaml, os, math

OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data"
R, NC = 1.0, 160
E1, E2, E3, G, nu = 37e9, 9e9, 9e9, 4e9, 0.3
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]

for hRs in HR:
    t = float(hRs) * R
    R_oml = R + t / 2.0
    ang = [2 * math.pi * k / NC for k in range(NC)]
    nodes = [[R_oml * math.cos(a), R_oml * math.sin(a), 0.0] for a in ang]
    elements = [[k + 1, (k + 1) % NC + 1] for k in range(NC)]
    oris = []
    for k in range(NC):
        tm = 2 * math.pi * (k + 0.5) / NC
        c, s = math.cos(tm), math.sin(tm)
        oris.append([0.0, 0.0, 1.0, -s, c, 0.0, -c, -s, 0.0])  # e1 axial, e2 tangent, e3 inward
    d = {
        "nodes": nodes,
        "elements": elements,
        "sections": [{"elementSet": "tube",
                      "layup": [["mat", t / 2.0, 45.0], ["mat", t / 2.0, -45.0]]}],  # outer +45, inner -45
        "sets": {"element": [{"name": "tube", "labels": list(range(1, NC + 1))}]},
        "materials": [{"name": "mat", "density": 1800.0,
                       "elastic": {"E": [E1, E2, E3], "G": [G, G, G], "nu": [nu, nu, nu]}}],
        "elementOrientations": oris,
    }
    with open(os.path.join(OUT, "shell_aniso_%s.yaml" % hRs), "w") as f:
        yaml.safe_dump(d, f, default_flow_style=False, sort_keys=True)
    print("wrote shell_aniso_%s.yaml  R_oml=%.4f  t=%.4f  [45/-45]" % (hRs, R_oml, t))
