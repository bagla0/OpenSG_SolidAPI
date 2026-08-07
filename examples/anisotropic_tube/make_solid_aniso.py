"""Create the [45/-45] tube 2D-solid annulus meshes with +45 OUTERMOST / -45 INNERMOST
(matching the shell, which stacks from the OML inward as [45, -45]). Fiber per element:
e1 = cos(phi)*axial + sin(phi)*tangent, e3 = outward radial; phi=+45 outer half, -45 inner.
Then DECODE the actual fiber angle of the innermost and outermost rings to verify."""
import os, math
import numpy as np

OUT = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\aniso_tube\solid_meshes"
os.makedirs(OUT, exist_ok=True)
ANI = dict(E=[37e9, 9e9, 9e9], G=[4e9, 4e9, 4e9], nu=[0.3, 0.3, 0.3])
R, NC, NR = 1.0, 200, 16
HR = [0.01, 0.03, 0.06, 0.12, 0.2, 0.35, 0.4]


def gen(path, H, verify=False):
    ri, ro = R - H / 2, R + H / 2
    rad = np.linspace(ri, ro, NR + 1)
    ang = np.array([2 * np.pi * k / NC for k in range(NC)])
    nid = lambda i, k: i * NC + (k % NC)
    nodes = [(rad[i] * np.cos(ang[k]), rad[i] * np.sin(ang[k])) for i in range(NR + 1) for k in range(NC)]
    elems, oris = [], []
    for i in range(NR):
        fr = (i + 0.5) / NR                       # 0 = inner, 1 = outer
        phi = math.radians(+45.0 if fr >= 0.5 else -45.0)   # OUTER half +45, INNER half -45
        cf, sf = math.cos(phi), math.sin(phi)
        for k in range(NC):
            a, b = nid(i, k), nid(i, k + 1)
            c, d = nid(i + 1, k + 1), nid(i + 1, k)
            elems.append((a + 1, b + 1, c + 1, d + 1))
            th = ang[k] + np.pi / NC              # element mid angle
            ct, st = np.cos(th), np.sin(th)
            # e1 = cos(phi)*axial(0,0,1) + sin(phi)*tangent(-st,ct,0); e3 = radial(ct,st,0); e2 = e3 x e1
            e1 = (sf * (-st), sf * ct, cf)
            e2 = (cf * (-st), cf * ct, -sf)
            e3 = (ct, st, 0.0)
            oris.append([e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2]])
    with open(path, "w") as f:
        f.write("nodes:\n")
        for y2, y3 in nodes:
            f.write(" - [%.10f %.10f 0.0]\n" % (y2, y3))
        f.write("elements:\n")
        for e in elems:
            f.write(" - [%d %d %d %d]\n" % e)
        f.write("sets:\n  element:\n  - name: tube\n    labels: [" + ", ".join(str(j + 1) for j in range(len(elems))) + "]\n")
        f.write("materials:\n - name: tube\n")
        f.write("   E: %s\n   G: %s\n   nu: %s\n   rho: 1800.0\n" % (ANI["E"], ANI["G"], ANI["nu"]))
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")
    if verify:
        def decode(eidx):
            o = oris[eidx]
            e1 = np.array(o[0:3]); k = eidx % NC
            th = ang[k] + np.pi / NC
            tan = np.array([-np.sin(th), np.cos(th), 0.0])
            return math.degrees(math.atan2(float(e1 @ tan), float(e1[2]))), math.hypot(*nodes[elems[eidx][0] - 1])
        ai, ri_ = decode(0)                         # innermost band, first element
        ao, ro_ = decode((NR - 1) * NC)             # outermost band, first element
        print("   VERIFY: innermost ring r~%.3f -> fiber %+.1f deg ;  outermost ring r~%.3f -> fiber %+.1f deg"
              % (ri_, ai, ro_, ao))


for hr in HR:
    p = os.path.join(OUT, "tube_aniso_hR%s.yaml" % hr)
    gen(p, hr * R, verify=(abs(hr - 0.06) < 1e-9))
    print("wrote %s  (ri=%.3f ro=%.3f, +45 outer / -45 inner)" % (os.path.basename(p), R - hr * R / 2, R + hr * R / 2))
