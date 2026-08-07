# Closed-tube EI2 ≠ EI3 anomaly — root cause

**Symptom.** FEniCS DG-penalty shell on the closed isotropic tube gives EI2 ≠ EI3
even though axisymmetry requires them equal. Cleanest at thin walls (h/R=0.01:
EI2 −10.3 %, EI3 −1.0 % vs solid). The warping-free `D_ee` is symmetric; the
warping term `D1 = V0ᵀ(−Dhe)` is asymmetric → the V0 fluctuation solve breaks the
90° rotational symmetry.

## Root cause: the C1-continuity penalty is too weak
The Kirchhoff shell needs C1 (slope) continuity for its 4th-order operator. The
FEniCS model enforces it **weakly**, by an interior penalty (`deri_constraint`,
`utils/shell.py:677`), with effective strength `alpha = 10·penalty/h_avg²` and
`penalty = 1e9` hardcoded (`core/shell.py:511`). At the default 1e9 the residual
slope-jump is non-negligible and is **anchored at the closed-loop seam (θ=0)**; it
differentially corrupts whichever bending mode aligns with the seam. The JAX
Hermite-C1 model enforces C1 *strongly* per node (no penalty) and keeps EI2==EI3.

## Evidence (drivers in this folder, iso tube h/R=0.01, center ref)
| test | driver | result | reading |
|---|---|---|---|
| **seam rotation** φ=0/45/90 | `make_rot_tube.py`+`run_rot.py` | gap −9.87 / +0.39 / +10.02 % | rides with the seam; EI2↔EI3 **swap** at 90°, **equal** at 45° → seam-localized |
| **mesh refinement** NC=160/320/640 | `make_refine_tube.py`+`run_refine.py` | gap −9.87 / +2.0 / +0.2 % | **converges away** (alpha ∝ 1/h²) |
| **penalty sweep** 1e6→1e13 | `run_reg.py` (OPENSG_PENALTY) | gap −9.87/−9.87/−3.05/−0.01 % | **1e13 → EI2==EI3** exactly (both −0.99 %) |
| consistency-term ablation | `run_ablation.py` (OPENSG_PURE_PENALTY) | no change (−9.98 %) | the n-linear Nitsche terms are **not** the cause |
| force CCW frame | (OPENSG_CCW_FRAME) | bit-identical | seam cell tangent is **not** sign-flipped |

## Fixes (trade-offs)
1. **Refine the cross-section** (NC ≥ 320–640): no code change, no regression; gap
   < 0.2 %, both EI within 1 % of solid. Cheapest correct fix for closed sections.
2. **Raise the penalty** to ~1e12–1e13 (`core/shell.py:511`): fixes EI2==EI3 and
   leaves airfoil EB stable (<0.2 %), **but** shifts airfoil Timoshenko shear GA2
   ~+20 % (st15 4.11e8→4.93e8) and ill-conditions EA near 1e13 → requires full
   st15/st12 re-validation before adopting globally.
3. **Strong nodal C1 (Hermite)** as in the JAX model — exact, but a larger rewrite.

## Note
EI2 ≠ EI3 is a *separate* defect from the thick-wall GJ/EI/GA collapse: at h/R=0.2
both EI stay ~−19 % even at penalty 1e13 — that is the genuine Kirchhoff thin-wall
breakdown, not the seam/penalty asymmetry.

## Fix applied (decoupled penalty) — `core/shell.py` `compute_timo_boun`
1. `penalty = 1e4 * max(|ABD entry|)` (≈1e13 here) replacing the hardcoded 1e9 —
   stiffness-scaled, restores tube **EI2==EI3** across the whole sweep
   (gap −0.15/−0.02/+0.03 % at h/R=0.01/0.06/0.2) and makes GA2==GA3.
2. A *global* high penalty **blows up the open-section shear GA2** (st15
   −8 %→+157 %, st12 −8 %→+94 % vs JAX → would fail the SHEAR_RTOL=12 % test),
   because the V1s/Timoshenko solve gets over-stiffened. So assemble a **second
   operator `A_l_v1` with the original 1e9 penalty and use it only in the V1s
   solve**, keeping the high penalty for the V0/EB solve.

Result (vs JAX): tube EI2==EI3 fixed; **st15/st12 GA2 back to −8.2 %** (passes),
EB within 2 %, GA3 slightly improved. Thick-wall under-prediction unchanged
(it is the thin-wall breakdown, not this anomaly). Not yet committed to OpenSG-1.0.

| st15 vs JAX | EA | GA2 | GA3 | GJ | EI2 | EI3 |
|---|---|---|---|---|---|---|
| penalty 1e9 (orig) | +0.03 | −8.0 | −8.5 | +0.95 | +0.49 | −0.10 |
| global high penalty | +0.35 | **+157** | −5.7 | +1.8 | +0.54 | −0.07 |
| decoupled (fix) | +0.35 | **−8.2** | −5.7 | +1.8 | +0.54 | −0.07 |
