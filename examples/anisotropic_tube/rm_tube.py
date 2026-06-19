"""Reissner-Mindlin (RM) MSG thin-walled shell -> Timoshenko 6x6, for a CLOSED
isotropic cylinder. Same RM machinery as strip_RM.py, but the wall curvature k22
is computed from the geometry (mesh_curvature) instead of the flat-strip k22=0."""
import os, sys
import numpy as np
from scipy.sparse import coo_matrix
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(HERE, "rm"))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pypardiso
from fe_jax import load_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml, mesh_curvature
from fe_jax.msg_materials import shift_abd_reference
from fe_jax.msg_solver import solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff
from msg_rm_timo import assemble_all, build_C_Psi
from transverse_shear import transverse_shear_stiffness


def rm_tube_6x6(yaml_path, frac, is_closed=True):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=is_closed))  # TUBE wall curvature ~ -1/R

    def D_of(i):
        abd = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(abd, frac * float(sum(i["thick"]))) if frac else abd
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_all(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    C, Psi = build_C_Psi(nodes2d, elems, p=1)
    Dc = C.T
    V0, D1, A_aug = solve_fluctuation_field(coo_matrix(Dhh), -Dhe, Dc)
    Deff = Dee + np.asarray(D1)
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
    n = Dhh.shape[0]
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n, :]), jnp.array(V0), jnp.array(Deff), V0DllV0, DhlV0, DhlTV0Dle,
        jnp.array(Psi), jnp.array(Dc))
    return np.asarray(C6), k22


if __name__ == "__main__":
    yaml = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\benchmark_tube\data\shell_iso_0.06.yaml"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    C6, k22 = rm_tube_6x6(yaml, frac)
    d = np.diag(C6)
    print("k22 (curvature) mean=%.4f  (expect ~ -1/R)" % float(np.mean(k22)))
    print("RM tube frac=%s: EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e"
          % (frac, d[0], d[1], d[2], d[3], d[4], d[5]))
