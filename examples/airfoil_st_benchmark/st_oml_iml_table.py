"""st15 + st12 Timoshenko 6x6 at OML / center / IML references, all non-zero terms,
JAX Kirchhoff + JAX RM vs the 2D-solid reference.  Plus a verification that the ABD
reference origin shifts correctly with frac and that e3 (ply normal) is inward, so the
frac shift is consistent with the material orientation.

Reference convention: frac=0 OML (outer face, z_ref=0), 0.5 center (mid-surface),
1.0 IML (inner face, z_ref=sum(thick)).  Mesh: offset_oml_to_iml along e3 (inward).
ABD: shift_abd_reference(z0=frac*thick) -- parallel-axis, e3 KEPT (no layup reversal).
"""
import sys, os
import numpy as np
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "rm"))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import read_mesh, mesh_curvature, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]
DATA = os.path.join(HERE, "training data", "opensg-FEniCS", "data")
ST = {
    "st15": dict(yaml=r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml",
                 solid=os.path.join(DATA, "bar_urc-15-t-0.in.K"), kind="vabs", zero_axial=False),
    "st12": dict(yaml=os.path.join(DATA, "1Dshell_12.yaml"),
                 solid=os.path.join(HERE, "outputs", "st12", "solid_C6_st12.txt"),
                 kind="txt", zero_axial=True),
}


def read_solid(cfg):
    if cfg["kind"] == "txt":
        return np.loadtxt(cfg["solid"])
    lines = open(cfg["solid"]).read().splitlines()
    for i, l in enumerate(lines):
        if "Timoshenko Stiffness Matrix" in l:
            rows = []
            for l2 in lines[i + 1:]:
                p = l2.split()
                if len(p) == 6:
                    try:
                        rows.append([float(x) for x in p])
                    except ValueError:
                        continue
                if len(rows) == 6:
                    break
            return np.array(rows)
    raise RuntimeError("VABS Timoshenko matrix not found")


def verify_abd_frac(name, cfg):
    """Show the ABD origin shifts with frac, the two shift mechanisms agree, e3 is inward."""
    n3d, elements, mat_db, layup_db, e2l = load_yaml(cfg["yaml"])
    ln = max(layup_db, key=lambda k: sum(layup_db[k]["thick"]))   # thickest layup -> clearest B
    i = layup_db[ln]; tot = float(sum(i["thick"]))
    A0 = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
    print("[%s] ABD frac/e3 verification on thickest layup '%s' (t=%.4g, %d plies)" % (
        name, ln, tot, len(i["thick"])))
    for frac in [0.0, 0.5, 1.0]:
        z = frac * tot
        Az = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db, z_ref=z)[0])
        As = shift_abd_reference(A0, z)                       # parallel-axis from OML
        Bz = Az[:3, 3:]
        ref = {0.0: "OML", 0.5: "center", 1.0: "IML"}[frac]
        print("   frac=%.1f (%-6s z_ref=%.4g): max|B|=%.3e   (mesh-z_ref vs parallel-axis shift) maxdiff=%.2e" % (
            frac, ref, z, np.max(np.abs(Bz)), np.max(np.abs(Az - As))))
    # e3 inward: IML mesh should sit inside the OML mesh (closer to centroid on average)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    e3 = element_e3_from_yaml(cfg["yaml"])
    iml = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=1.0)
    cen = nodes[:, :2].mean(axis=0)
    d_o = np.linalg.norm(nodes[:, :2] - cen, axis=1).mean()
    d_i = np.linalg.norm(iml[:, :2] - cen, axis=1).mean()
    print("   e3 inward check: mean |node-centroid|  OML=%.4g  IML=%.4g  -> %s" % (
        d_o, d_i, "INWARD ok" if d_i < d_o else "*** OUTWARD?! ***"))


def shell_6x6(cfg, frac):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(cfg["yaml"])
    if cfg["zero_axial"]:
        n3d = n3d.copy(); n3d[:, 2] = 0.0
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(cfg["yaml"])
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac * float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(cfg["yaml"], frac=frac)
    return np.asarray(RM), np.asarray(KF)


def table(name, cfg):
    S = read_solid(cfg)
    res = {f: shell_6x6(cfg, f) for f in (0.0, 0.5, 1.0)}      # (RM, KF) per frac
    print("\n================= %s : Timoshenko 6x6, %% error vs 2D solid =================" % name)
    print("order [ext,shear2,shear3,twist,bend2,bend3]; non-zero terms (|solid|>1e5)")
    print("%-11s %12s | %-21s | %-21s" % ("term", "solid", "Kirchhoff  OML/cen/IML",
                                          "RM        OML/cen/IML"))
    for a in range(6):
        for b in range(a, 6):
            v = S[a, b]
            if abs(v) < 1e5:
                continue
            nm = "%s-%s" % (LBL[a][:3], LBL[b][:3])
            kf = "/".join("%+6.1f" % (100 * (res[f][1][a, b] - v) / v) for f in (0.0, 0.5, 1.0))
            rm = "/".join("%+6.1f" % (100 * (res[f][0][a, b] - v) / v) for f in (0.0, 0.5, 1.0))
            print("%-2d,%-2d %-5s %12.3e | %-21s | %-21s" % (a + 1, b + 1, nm, v, kf, rm))


if __name__ == "__main__":
    for name, cfg in ST.items():
        verify_abd_frac(name, cfg)
    for name, cfg in ST.items():
        table(name, cfg)
