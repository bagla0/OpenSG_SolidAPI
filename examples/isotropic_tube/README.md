# Isotropic cylinder — thickness sweep, % error vs FEniCS solid

> **Archive of the isotropic-tube cross-validation study** (meshes + drivers + results +
> plot). It cross-checks the OpenSG FEniCS Kirchhoff shell and 2D solid against the JAX
> thin-walled toolkit and the MSG thin-wall analytical, over h/R = 0.01 → 0.5.
>
> **Environments:** the FEniCS drivers (`sweep_shell.py`, `sweep_solid.py`) run in the
> dolfinx env; the JAX drivers (`sweep_jax.py`, `rm_tube.py`, `make_*.py`) use the separate
> JAX package (`opensg_jax` / `rm`). Scripts retain the absolute paths they were run with —
> adjust `PKG`/`HERE` to your checkout.
>
> **Note on the FEniCS-shell results:** the near-0 % center-reference curves were produced
> with three small, localized improvements to `opensg/core/shell.py` + `opensg/mesh/segment.py`
> (none architectural): (1) a `frac` argument on `compute_ABD_matrix`/`compute_ABD` selecting
> the through-thickness reference (frac=0 OML default, 0.5 mid-surface); (2) the C1 penalty
> scaled to `2e2·max(E)` instead of a fixed constant; (3) the V1s/shear solve kept at the
> original penalty so open-section shear is unaffected. These are proposed separately; on
> stock `main` the FEniCS-shell center curve under-predicts at thick h/R (see `debug_ei2/`).

Isotropic circular cylinder, mid-wall radius **R = 1.0 m**, **E = 70 GPa, ν = 0.3**
(G = 26.923 GPa). Wall thickness swept **h/R = 0.01, 0.03, 0.06, 0.12, 0.2, 0.35, 0.5**
(thin → very thick). Each shell method is run at **two references**:

* **center** — shell line on the wall mid-surface (R_mid = 1.0); matches the solid centroid
* **OML** — shell line on the outer surface (R_oml = R + t/2); over-states the line radius

The **FEniCS 2D solid** (full annulus) is the ground truth; everything is plotted as
**% error vs the solid** for all six Timoshenko diagonal terms `[EA, GA2, GA3, GJ, EI2, EI3]`.

## Methods / drivers
| method (legend) | code | driver | reference knob |
|---|---|---|---|
| JAX Kirchhoff (Hermite C1) | `opensg_jax/fe_jax` | `sweep_jax.py` | `frac=0.5` center / `frac=0.0` OML |
| JAX RM | `Claude_code/rm` | `rm_tube.py`+`sweep_jax.py` | same `frac` |
| FEniCS Kirchhoff (DG penalty) | `opensg-FEniCS` | `sweep_shell.py` | `1Dtube_iso_*` center / `1Dtube_iso_OML_*` |
| FEniCS solid (reference) | `opensg-FEniCS` | `sweep_solid.py` | annulus geometry |

## Pipeline
1. `make_shell_iso.py` → JAX tube shells `shell_iso_<hR>.yaml` (line at OML, ply inward).
2. `make_solid_extra.py` → thick annuli `tube_iso_hR{0.35,0.5}.yaml` (extends 0.01–0.2 set).
3. `make_tube_mesh.py` → FEniCS center `1Dtube_iso_<hR>.yaml` + OML `1Dtube_iso_OML_<hR>.yaml`.
4. `sweep_jax.py` (Windows `opensg_2_0_env`) → `results/jax.txt`.
5. `sweep_solid.py`, `sweep_shell.py` (WSL `opensg_env_v8`, run from a no-space copy) → `results/{solid,shell}.txt`.
6. `compile_plot.py` → `cylinder_pcterr.png` + the % error tables.

## Result — % error vs solid (CENTER reference)
```
            h/R=0.01  0.06   0.12   0.2    0.35   0.5
JAX-Kirch EA   +0.01  +0.01  +0.02  +0.04  +0.10  +0.19
JAX-Kirch GA2  -0.01  -0.11  -0.39  -1.05  -3.04  -5.75
JAX-Kirch GJ   -0.01  +0.02  +0.11  +0.32  +0.98  +1.95
JAX-Kirch EI3  -0.00  -0.06  -0.23  -0.63  -1.89  -3.76
JAX-RM    GA2  -0.00  -0.08  -0.31  -0.85  -2.48  -4.71   (shear-corrected, best)
JAX-RM    EI3  -0.00  -0.05  -0.19  -0.53  -1.61  -3.21
shell-DG  GA2  -1.53  -8.77 -16.95 -26.99 -43.08 -55.69   <-- collapses
shell-DG  GJ   -2.49 -14.30 -27.11 -41.81 -61.80 -71.91   <-- collapses
shell-DG  EI3  -1.00  -5.94 -11.75 -19.26 -32.46 -44.30
shell-DG  EI2 -10.31 -11.54 -13.82 -19.44 -32.61 -44.63   <-- EI2!=EI3 from thinnest
```

## Findings
1. **Center is the correct reference.** With the line on the mid-surface, JAX Kirchhoff
   matches the solid to <0.1 % up to h/R≈0.1 and ≤6 % even at h/R=0.5; **JAX RM is best**
   (EA exact, GA2 −4.7 %, EI −3.2 % at h/R=0.5 — the transverse-shear term recovers part of
   the thick-wall correction).
2. **OML reference adds a geometric offset that grows ≈ (h/R)/2.** Putting the line on the
   outer surface inflates EA and EI by ≈ +25 % and deflates GJ by ≈ −24 % at h/R=0.5 for
   *every* method (the curves shift bodily). This confirms the center line is the right choice
   for matching the solid centroid.
3. **The FEniCS Kirchhoff DG-penalty shell now tracks the solid at the center reference**
   (after the two fixes below): center GA2 −0.04→−3.6 %, GJ −0.01→+7.0 %, EI −0.05→−3.7 %
   over h/R 0.01→0.5 — comparable to JAX Kirchhoff, and far better than OML (the dashed
   curves carry the +(h/R)/2 geometric offset). The earlier −44 %/−72 % "collapse" was a
   reference bug, not a thin-wall limit.

### Two FEniCS-shell fixes (`opensg/core/shell.py`, `opensg/mesh/segment.py`)
4. **ABD reference (`frac`).** The through-thickness reference-centering in
   `compute_ABD_matrix` was commented out (z=0 at the ply OML surface), so a center-line
   mesh placed the wall t/2 too far in → the collapse. Added a JAX-style `frac` arg
   (default 0 = OML; 0.5 = mid-ply centroid), plumbed through `ShellBounMesh.compute_ABD`.
   `sweep_shell` uses frac=0.5 (center) / frac=0 (OML); st15/st12 use the default → unchanged.
5. **Penalty scaling.** The C1 penalty is `2e2·max(E)` (max Young's modulus, ~1.4e13 for
   E=70 GPa), injected via `meshdata["max_E"]`. Scaling to max(E) is thickness-independent,
   so it neither under-enforces (old fixed 1e9 → EI2≠EI3) nor explodes (max|ABD|∝E·t over-shot
   to ~4e13 and re-broke EI2==EI3 by ill-conditioning); the clean window is ~[1e13, 3e13].
   The V1s/shear solve keeps a separate 1e9 penalty so the open-section airfoil GA2
   (st15/st12, −8 % vs JAX) is preserved (a global high penalty blew GA2 to +157 %).
   EI2==EI3 now holds across the whole sweep (gap ≤0.8 %). See `debug_ei2/FINDINGS.md`.

6. **OML's +20-25 % is geometric, not a bug.** OML/center ratios equal R_oml/R_mid to 4
   sig-figs (EA 1.2547 vs 1.25 at h/R=0.5), and JAX shows the identical scaling — so the
   material orientation / ABD d3 are correct; the offset is the thin-shell reference-surface
   effect (section integrated at the line radius; the (1+z·κ) curved-shell metric is omitted).
   Center (line on the wall mid-surface) is the correct reference.
7. **MSG thin-wall analytical** (EA=E·2πRt, GA=½G·2πRt, GJ=G·2πR³t, EI=E·πR³t) added to the
   plot as the dotted baseline: EA exact, GJ/EI −5.85 %, GA −9.1 % at h/R=0.5 (the dropped
   O((t/R)²)). The numerical shells (FEniCS/JAX center) sit *closer* to the solid than this
   leading-order analytical at thick walls, because they carry the curvature (k22) terms.
   The FEniCS **solid is the ground-truth baseline** (its mesh origin is always the geometric
   origin = beam axis, so it has no center/OML choice — not a "ref=0").

See `cylinder_pcterr.png` (2×3 panels, solid line = center, dashed = OML, dotted = analytical).
`cylinder_sweep.png` (absolute values, center only, h/R≤0.2) is retained from the prior run.
```
results/  jax.txt  solid.txt  shell.txt     # raw 6x6 diagonals per tag/h/R
meshes/   shell_iso_*  1Dtube_iso_*(_OML)  tube_iso_hR*   # all meshes used
```
