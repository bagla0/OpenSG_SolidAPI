"""Convert the JAX [45/-45] tube shells (shell_aniso_hR) to FEniCS ShellBounMesh
format at center / OML / IML references (line scaled to R_mid / R_oml / R_iml).
Same convert() as the iso pipeline; layup [45,-45] (outer +45, inner -45) preserved."""
import yaml, os, math

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data"
OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.4"]


def convert(src, scale):
    L = ["nodes:"]
    for n in src["nodes"]:
        L.append("  - [%s %s %s]" % (n[0] * scale, n[1] * scale, n[2]))
    L.append("elements:")
    for e in src["elements"]:
        L.append("  - [%d %d]" % (int(e[0]), int(e[1])))
    L.append("sections:")
    for sec in src["sections"]:
        L.append("- type: shell"); L.append("  elementSet: %s" % sec["elementSet"]); L.append("  layup:")
        for ply in sec["layup"]:
            L.append("  - - %s" % ply[0]); L.append("    - %s" % ply[1]); L.append("    - %s" % ply[2])
    L.append("sets:"); L.append("  element:")
    for st in src["sets"]["element"]:
        L.append("  - name: %s" % st["name"]); L.append("    labels:")
        for lab in st["labels"]:
            L.append("    - %d" % int(lab))
    L.append("materials:")
    for mat in src["materials"]:
        L.append("- name: %s" % mat.get("name", "mat")); L.append("  density: %s" % mat.get("density", 1.0))
        el = mat["elastic"]; L.append("  elastic:")
        for key in ("E", "G", "nu"):
            L.append("    %s:" % key)
            for v in el[key]:
                L.append("    - %s" % v)
    L.append("elementOrientations:")
    for o in src["elementOrientations"]:
        L.append("- - %s" % o[0])
        for c in o[1:]:
            L.append("  - %s" % c)
    return "\n".join(L) + "\n"


for hR in HR:
    src = yaml.safe_load(open(os.path.join(DATA, "shell_aniso_%s.yaml" % hR)))
    R_oml = math.hypot(src["nodes"][0][0], src["nodes"][0][1])
    open(os.path.join(OUT, "1Dtube_aniso_%s.yaml" % hR), "w").write(convert(src, 1.0 / R_oml))   # center
    open(os.path.join(OUT, "1Dtube_aniso_OML_%s.yaml" % hR), "w").write(convert(src, 1.0))         # OML
    print("wrote 1Dtube_aniso_%s center+OML (R_oml=%.4f)" % (hR, R_oml))
