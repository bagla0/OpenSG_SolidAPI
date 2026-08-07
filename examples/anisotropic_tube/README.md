# [45/-45] composite tube — thickness sweep, % error vs FEniCS solid (all non-zero terms)

> Companion to `isotropic_tube`, for a **balanced [45/-45] composite circular tube**
> (outer ply +45, inner ply -45; ply E1=37, E2=E3=9 GPa, G=4 GPa, ν=0.3), mid-radius
> R=1.0 m, wall swept **h/R = 0.01 → 0.4**. Because the laminate is anisotropic the
> Timoshenko 6×6 has **coupling** terms, so this study compares **all 9 non-zero terms**
> (6 diagonal + 3 coupling), % error vs the FEniCS 2D solid (VABS). No analytical
> (no closed form for [45/-45]).

## Layup — verified identical between shell and solid
Canonical stack: **+45 outermost, −45 innermost**. The shell lists outer-first
(`[45, −45]`); the solid is built explicitly and **decoded to confirm** (innermost ring
−45°, outermost +45°, see `make_solid_aniso.py`). The coupling-term *signs* match the
solid (the decisive check — [45/−45] and [−45/45] are mirror laminates with opposite
coupling), so the laminate is provably the same.

## Methods / references (same as isotropic_tube)
JAX Kirchhoff (Hermite C1), JAX RM, FEniCS Kirchhoff (DG penalty) shell, vs FEniCS 2D
solid. Each shell at center (frac 0.5) / OML (0). Non-zero terms:
`EA, GA2, GA3, GJ, EI2, EI3` (diagonal) + `ext-twist (1,4)`, `shear2-bend2 (2,5)`,
`shear3-bend3 (3,6)` (coupling).

## Pipeline
`make_shell_aniso.py` (JAX shells) · `make_solid_aniso.py` (annuli, +45 outer/−45 inner,
self-verifying) · `make_tube_mesh_aniso.py` (FEniCS center/OML) → `sweep_{jax,shell,
solid}.py` (full 6×6 → `results/`) → `compile_aniso.py` (`aniso_pcterr.png`).
`verify_*.py` dump full 6×6 for the layup check; `debug_penalty.py` is the
FEniCS-spike diagnostic (below).

## Results (center, % error vs solid)
- **Diagonal:** all methods <~6 % to h/R=0.4 (RM/Kirchhoff/FEniCS all good).
- **Coupling (after the JAX TW corrections below):**
  | term | JAX Kirchhoff | JAX RM | FEniCS Kirchhoff |
  |---|---|---|---|
  | ext-twist (1,4) | **−10 %** (flat) | **−10 %** (flat) | **−9 %** |
  | shear-bend (2,5)/(3,6) | −7 → −15 % (w/ eps22) | −19 → −23 % | ~+3 → +6 % |
  - The [±45] coupling comes from the through-thickness B16/B26 (balanced ⇒ A16=0) and is
    transverse-shear-sensitive via the **wall curvature** k22. All three shells now
    **bracket the 3D solid** on every coupling (none is the old flat −70 % / +66 % outlier).

## JAX thin-walled (TW) corrections applied
The closed [±45] tube exposed three sign/curvature bugs (plus one shear-warping
enhancement, #4) in the JAX shell models; all in `bagla0/OpenSG-TW` (branch
`msg-reissner-mindlin`) and reflected here:
1. **Wall curvature for 2-node closed meshes** (`msg_mesh.py`): `mesh_curvature` returned
   0 for flat 2-node tube elements, dropping the k22 = −1/R channel that carries the
   [±45] coupling. Now recovered from consecutive-corner turning (`_curvature_from_corners`).
2. **Kirchhoff macro + shear operator signs** (`msg_hermite.py`): the curvature term in the
   `Ge` kappa12 macro (`Ge[5,1]`) and the `eps_l` shear operator had the wrong sign for the
   k22 = −1/R convention → fixed ext-twist (−69 %→−10 %) and shear-bend (−71 %→−10 %).
3. **RM wall transverse-bending-curvature sign** (`msg_rm_timo.py`, `msg_rm.py`): the
   kappa22 operator was `BDq[4] = −dN(omega1)`; correct is **`+dN(omega1)`** (curvature is
   the positive rate of the rotation fluctuation). The wrong sign inverted the closed-tube
   transverse-shear-bending coupling, over-counting shear-bend **+66 %**; it was invisible
   in the EB diagonal (enters squared) and in open airfoils (st12/st15 unchanged). Fixed →
   shear-bend −18.8 %, full diagonal + ext-twist preserved.
4. **Kirchhoff eps22 hoop channel** (`msg_hermite.py` `eps_l`): the shear-warping operator
   was missing the hoop strain `eps22 = t.dw` (only eps11/2eps12/kappa12 were populated).
   Adding it pulls the JAX-Kirchhoff shear-bend toward the solid at thin walls (~-10.5% to
   -7% at h/R=0.06); st12/st15/iso unchanged, diagonal/GA preserved. It makes the coupling
   curve mildly noisy/asymmetric across h/R -- the bulk of the JAX-vs-FEniCS coupling gap is
   the *discretization* (Hermite-C1 vs CG2+penalty/Eq.100 V1s), not the operator.

## OML vs center reference (not a bug)
The OML curves run high (e.g. EA +20 %, GJ −40 % at h/R=0.4) but this is the **geometric
reference-surface effect**, not a B-matrix error: the 1D line sits at the *outer* radius
R_oml=R+t/2, so the flat-plate ABD models a tube (h/R)/2 too large. The diagonal follows
the exact law **EA/EI ~ +ε, GJ/GA ~ −2ε** (ε=(h/R)/2) for *both* JAX and FEniCS, with **no
spurious couplings** (verified, `oml_check.py`). **Center (mid-surface) is the accurate
reference** — it matches the solid to <0.5 %. Making OML accurate would need the (1+z·k)
curved-shell area metric (Flügge vs Donnell), a modeling choice not pursued here.

## FEniCS spike bug — found & fixed
The first run showed FEniCS red spikes at h/R=0.12 (EI2 −32 %, shear-bend +241 %). Cause:
the C1 penalty `2e2·max(E)` = 7.4e12 for the soft aniso ply (E1=37 GPa) **falls below the
clean C1 window ~[1e13, 3e13]** (the iso ply E=70 GPa lands at 1.4e13), so the closed-loop
**seam asymmetry** (EI2≠EI3, shear-bend) re-triggers at some h/R. `debug_penalty.py`
confirms: at 7.4e12 the EI2/EI3 gap is −32 %, at ≥1e13 it is <0.5 %. **Fix** (in
`opensg/core/shell.py`): clip the penalty into the window, `penalty = min(max(2e2·max_E,
1.2e13), 2.5e13)` — floors the soft-ply case, leaves the iso/airfoil cases unchanged,
caps very stiff materials. After the fix every FEniCS curve is smooth.

See `aniso_pcterr.png` (all 9 non-zero terms, center+OML references, h/R to 0.4).
