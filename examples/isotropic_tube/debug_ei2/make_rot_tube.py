"""SEAM-ROTATION test meshes for the EI2!=EI3 closed-loop anomaly.
Build the iso tube (h/R=0.01, center ref R_mid=1) with the whole section rigidly
rotated by phi = 0, 45, 90 deg. Numbering is preserved, so the closure seam
(node NC <-> node 1) sits at global angle ~phi while the bending axes (global
x[1], x[2]) stay fixed. Writes FEniCS ShellBounMesh format (center reference)."""
import os, math

OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data"
R, NC = 1.0, 160
hr = 0.01
t = hr * R
R_oml = R + t / 2.0
E, G, nu = 70e9, 26.923e9, 0.3


def write_fenics(path, phi):
    s = 1.0 / R_oml  # scale OML circle -> R_mid = 1.0 (center reference)
    L = ["nodes:"]
    for k in range(NC):
        th = phi + 2 * math.pi * k / NC
        x, y = R_oml * math.cos(th), R_oml * math.sin(th)
        L.append("  - [%s %s %s]" % (x * s, y * s, 0.0))
    L.append("elements:")
    for k in range(NC):
        L.append("  - [%d %d]" % (k + 1, (k + 1) % NC + 1))
    L.append("sections:")
    L.append("- type: shell"); L.append("  elementSet: tube"); L.append("  layup:")
    L.append("  - - mat"); L.append("    - %s" % t); L.append("    - 0.0")
    L.append("sets:"); L.append("  element:")
    L.append("  - name: tube"); L.append("    labels:")
    for k in range(NC):
        L.append("    - %d" % (k + 1))
    L.append("materials:")
    L.append("- name: mat"); L.append("  density: 1800.0"); L.append("  elastic:")
    for key, v in (("E", E), ("G", G), ("nu", nu)):
        L.append("    %s:" % key)
        for _ in range(3):
            L.append("    - %s" % v)
    L.append("elementOrientations:")
    for k in range(NC):
        thm = phi + 2 * math.pi * (k + 0.5) / NC
        c, sn = math.cos(thm), math.sin(thm)
        o = [0.0, 0.0, 1.0, -sn, c, 0.0, -c, -sn, 0.0]
        L.append("- - %s" % o[0])
        for cc in o[1:]:
            L.append("  - %s" % cc)
    open(path, "w").write("\n".join(L) + "\n")


for phi_deg in [0, 45, 90]:
    p = os.path.join(OUT, "1Drot_iso_%d.yaml" % phi_deg)
    write_fenics(p, math.radians(phi_deg))
    print("wrote 1Drot_iso_%d.yaml  (seam at global theta~%d deg)" % (phi_deg, phi_deg))
