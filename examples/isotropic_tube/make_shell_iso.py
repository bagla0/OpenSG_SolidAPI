"""Generate JAX-format isotropic-tube shell meshes shell_iso_hR.yaml (line at the
OML, R_oml = R_mid + t/2; ply runs inward; e3 = inward normal). One mesh per h/R;
JAX picks the reference at run time via frac (frac=0 -> OML, frac=0.5 -> mid-wall).
Reproduces the format/orientation of the validated shell_iso_0.06.yaml exactly."""
import yaml, os, math

OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data"
R, NC = 1.0, 160
E, G, nu = 70e9, 26.923e9, 0.3
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]

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
        "sections": [{"elementSet": "tube", "layup": [["mat", t, 0.0]]}],
        "sets": {"element": [{"name": "tube", "labels": list(range(1, NC + 1))}]},
        "materials": [{"name": "mat", "density": 1800.0,
                       "elastic": {"E": [E, E, E], "G": [G, G, G], "nu": [nu, nu, nu]}}],
        "elementOrientations": oris,
    }
    with open(os.path.join(OUT, "shell_iso_%s.yaml" % hRs), "w") as f:
        yaml.safe_dump(d, f, default_flow_style=False, sort_keys=True)
    print("wrote shell_iso_%s.yaml  R_oml=%.4f  t=%.4f" % (hRs, R_oml, t))
