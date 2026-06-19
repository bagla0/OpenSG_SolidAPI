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
- **Diagonal:** all methods <~6 % to h/R=0.5 (RM best; FEniCS GJ +5.7 % at 0.5).
- **Coupling — the headline:**
  | term | JAX Kirchhoff | JAX RM | FEniCS Kirchhoff |
  |---|---|---|---|
  | ext-twist (1,4) | **−69 %** (flat) | −28 % | **−9 %** |
  | shear-bend (2,5)/(3,6) | **−71 %** (flat) | +56 % | ~+5 % |
  - The [±45] coupling comes from the through-thickness B16/B26 (balanced ⇒ A16=0) and is
    **transverse-shear-dominated**: JAX **Kirchhoff under-predicts every coupling by a flat
    ~70 %** (the B-mapping deficit), **RM recovers most** of ext-twist, and the **FEniCS
    Kirchhoff shell matches the solid to ~10 %** on all couplings.

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
