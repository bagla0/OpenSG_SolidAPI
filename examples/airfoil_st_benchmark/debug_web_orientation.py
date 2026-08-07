"""Debug: compare the material frame (e2 tangent, e3 ply-normal) on the WEB between the
thin-walled (1D shell) and the solid (2D) models.  The web is the internal ~vertical
member; identify its elements by a vertical tangent and compare e2/e3 directions."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from plot_sections import load, ST


def main(name):
    cfg = ST[name]
    print("==================== %s ====================" % name)

    # ---- thin-walled: web = elements with near-vertical tangent ----
    n, el, ori = load(cfg["shell"]); xy = n[:, :2]
    web = []
    for i, e in enumerate(el):
        t = xy[e[-1] - 1] - xy[e[0] - 1]
        if abs(t[1]) > 2.0 * abs(t[0]) and np.linalg.norm(t) > 1e-9:
            web.append(i)
    print("THIN-WALLED: %d web (vertical-tangent) elements" % len(web))
    wx = []
    for i in web:
        e = el[i]; m = 0.5 * (xy[e[0] - 1] + xy[e[-1] - 1])
        e2 = np.array(ori[i][3:5]); e3 = np.array(ori[i][6:8]); wx.append(m[0])
        print("   x=%+.3f y=%+.3f  e2=(%+.3f,%+.3f)  e3=(%+.3f,%+.3f)  [e3_x sign %s]" % (
            m[0], m[1], e2[0], e2[1], e3[0], e3[1], "+" if e3[0] > 0 else "-"))
    wx = np.array(wx)

    # ---- solid: elements whose centroid sits near the web x-columns ----
    ns, els, oris = load(cfg["solid"]); sxy = ns[:, :2]
    cols = sorted(set(round(x, 1) for x in wx))
    print("SOLID: web x-columns ~", cols)
    for xc in cols:
        sel = []
        for j, e in enumerate(els):
            c = sxy[[k - 1 for k in e]].mean(axis=0)
            if abs(c[0] - xc) < 0.06 and abs(oris[j][4]) > abs(oris[j][3]):  # vertical e2 = true web
                sel.append((j, c))
        if not sel:
            continue
        e3x = np.array([oris[j][6] for j, _ in sel])
        e2 = np.array([oris[j][3:5] for j, _ in sel]).mean(axis=0)
        e3 = np.array([oris[j][6:8] for j, _ in sel]).mean(axis=0)
        sgn = "+x" if e3[0] > 0 else "-x"
        print("   x~%+.1f: %d WEB elem (vertical e2)  mean e2=(%+.3f,%+.3f)  mean e3=(%+.3f,%+.3f) -> e3 %s  [%d(+x) %d(-x)]" % (
            xc, len(sel), e2[0], e2[1], e3[0], e3[1], sgn, int((e3x > 0).sum()), int((e3x < 0).sum())))


if __name__ == "__main__":
    for nm in ("st15", "st12"):
        main(nm)
