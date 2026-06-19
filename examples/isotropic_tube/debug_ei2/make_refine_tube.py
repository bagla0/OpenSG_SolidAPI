"""Refinement test: iso tube h/R=0.01, seam at phi=0, center ref, NC = 160,320,640.
If the EI2-EI3 gap SHRINKS with refinement -> weak/consistent seam C1 penalty
(converges away). If it STAYS -> a structural topology defect (missing seam facet)."""
import os, math
OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
R = 1.0
hr = 0.01
t = hr * R
R_oml = R + t / 2.0
E, G, nu = 70e9, 26.923e9, 0.3


def write_fenics(path, NC):
    s = 1.0 / R_oml
    L = ["nodes:"]
    for k in range(NC):
        th = 2 * math.pi * k / NC
        L.append("  - [%s %s %s]" % (R_oml * math.cos(th) * s, R_oml * math.sin(th) * s, 0.0))
    L.append("elements:")
    for k in range(NC):
        L.append("  - [%d %d]" % (k + 1, (k + 1) % NC + 1))
    L.append("sections:"); L.append("- type: shell"); L.append("  elementSet: tube"); L.append("  layup:")
    L.append("  - - mat"); L.append("    - %s" % t); L.append("    - 0.0")
    L.append("sets:"); L.append("  element:"); L.append("  - name: tube"); L.append("    labels:")
    for k in range(NC):
        L.append("    - %d" % (k + 1))
    L.append("materials:"); L.append("- name: mat"); L.append("  density: 1800.0"); L.append("  elastic:")
    for key, v in (("E", E), ("G", G), ("nu", nu)):
        L.append("    %s:" % key)
        for _ in range(3):
            L.append("    - %s" % v)
    L.append("elementOrientations:")
    for k in range(NC):
        thm = 2 * math.pi * (k + 0.5) / NC
        c, sn = math.cos(thm), math.sin(thm)
        o = [0.0, 0.0, 1.0, -sn, c, 0.0, -c, -sn, 0.0]
        L.append("- - %s" % o[0])
        for cc in o[1:]:
            L.append("  - %s" % cc)
    open(path, "w").write("\n".join(L) + "\n")


for NC in [160, 320, 640]:
    write_fenics(os.path.join(OUT, "1Dref_iso_%d.yaml" % NC), NC)
    print("wrote 1Dref_iso_%d.yaml" % NC)
