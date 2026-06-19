"""Convert the JAX-format iso-tube shells (shell_iso_hR) to FEniCS ShellBounMesh
format (space-separated string nodes, type: shell). For each h/R writes TWO meshes:

  1Dtube_iso_<hR>.yaml       CENTER reference: circle scaled R_oml -> R_mid = 1.0
                             (the shell line sits on the wall mid-surface)
  1Dtube_iso_OML_<hR>.yaml   OML reference: circle kept at R_oml = R_mid + t/2
                             (the shell line sits on the outer surface)

The center mesh removes the line-radius offset (~(h/R)/2 in EA, ~3(h/R)/2 in EI)
that the OML reference carries; comparing the two isolates reference sensitivity."""
import yaml, os, math

DATA = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data"
OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
HR = ["0.01", "0.03", "0.06", "0.12", "0.2", "0.35", "0.5"]


def convert(src, scale):
    nodes = src["nodes"]; elements = src["elements"]
    sections = src["sections"]; sets = src["sets"]; mats = src["materials"]
    orient = src["elementOrientations"]
    L = ["nodes:"]
    for n in nodes:
        L.append("  - [%s %s %s]" % (n[0] * scale, n[1] * scale, n[2]))
    L.append("elements:")
    for e in elements:
        L.append("  - [%d %d]" % (int(e[0]), int(e[1])))
    L.append("sections:")
    for sec in sections:
        L.append("- type: shell"); L.append("  elementSet: %s" % sec["elementSet"]); L.append("  layup:")
        for ply in sec["layup"]:
            L.append("  - - %s" % ply[0]); L.append("    - %s" % ply[1]); L.append("    - %s" % ply[2])
    L.append("sets:"); L.append("  element:")
    for st in sets["element"]:
        L.append("  - name: %s" % st["name"]); L.append("    labels:")
        for lab in st["labels"]:
            L.append("    - %d" % int(lab))
    L.append("materials:")
    for mat in mats:
        L.append("- name: %s" % mat.get("name", "mat")); L.append("  density: %s" % mat.get("density", 1.0))
        el = mat["elastic"]; L.append("  elastic:")
        for key in ("E", "G", "nu"):
            L.append("    %s:" % key)
            for v in el[key]:
                L.append("    - %s" % v)
    L.append("elementOrientations:")
    for o in orient:
        L.append("- - %s" % o[0])
        for c in o[1:]:
            L.append("  - %s" % c)
    return "\n".join(L) + "\n"


for hR in HR:
    src = yaml.safe_load(open(os.path.join(DATA, "shell_iso_%s.yaml" % hR)))
    R_oml = math.hypot(src["nodes"][0][0], src["nodes"][0][1])
    open(os.path.join(OUT, "1Dtube_iso_%s.yaml" % hR), "w").write(convert(src, 1.0 / R_oml))      # center
    open(os.path.join(OUT, "1Dtube_iso_OML_%s.yaml" % hR), "w").write(convert(src, 1.0))           # OML
    print("wrote 1Dtube_iso_%s (center R_mid=1.0) + _OML_%s (R_oml=%.4f)" % (hR, hR, R_oml))
