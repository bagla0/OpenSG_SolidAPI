# st15 / st12 thin-walled vs 2D-solid benchmark (OML / center / IML)

Timoshenko 6×6 stiffness of two wind-turbine-blade airfoil stations (**st15**, **st12**)
computed with the JAX MSG thin-walled shell models (**Kirchhoff** Hermite-C1 and
**Reissner-Mindlin**) at three through-thickness reference surfaces — **OML** (outer,
frac 0), **center** (mid-surface, frac 0.5), **IML** (inner, frac 1.0) — and compared
term-by-term to the **2D solid** (st15: VABS; st12: FEniCSx solid).

## Files
| file | what |
|---|---|
| `st_oml_iml_table.py` | driver: the ABD frac/e3 **verification** + the text tables (all non-zero terms, OML/center/IML, Kirchhoff + RM vs solid) |
| `plot_sections.py` | cross-section figures: thin-walled line + solid 2D mesh, each with the material frame **e1 (beam axis, out of page), e2 (tangent), e3 (inward normal)** |
| `make_table_png.py` | renders the tables as colour-coded PNGs |
| `st15_sections.png`, `st12_sections.png` | cross-sections + e1/e2/e3 |
| `st15_table.png`, `st12_table.png` | the % error tables (green <5 %, yellow <15 %, red >15 %) |

Data: st15 `1Dshell_15.yaml` + `2Dsolid_VABS_15.yaml`; st12 `1Dshell_12.yaml` +
`2Dsolid_12.yaml` (all in `opensg-FEniCS/data` except the st15 shell).

## e3 / frac / ABD-origin verification (this is the key correctness check)
The reference change must move the **ABD origin through the thickness as frac changes**,
with e3 (the ply-stacking normal) kept consistent with the material orientation — **not**
a layup reversal. Verified numerically:

```
                max|B|  OML / center / IML        z_ref(mesh) vs shift_abd_reference   e3 inward
st15 (t=0.076)  5.63e6 / 2.29e6 / 1.02e7          identical to ~1e-7                   OML 1.021 > IML 0.994  ok
st12 (t=0.084)  9.07e6 / 2.49e6 / 1.41e7          identical to ~1e-7                   OML 1.196 > IML 1.177  ok
```
- The bending-extension **B block changes with frac** and is **minimised at center** (the
  parallel-axis part cancels, leaving only the intrinsic laminate B). The origin really
  moves OML→IML.
- The two ways to shift the reference agree to ~1e-7: `compute_ABD_matrix(z_ref=frac·t)`
  (mesh-origin shift) and `shift_abd_reference(ABD, frac·t)` (parallel-axis). The latter
  **keeps e3 fixed** (no layup flip) — the concern in the `check-e3-orientation` skill.
- e3 (= YAML `elementOrientations` o[6],o[7]) points **OML→IML**; the ply stack runs
  z=0(OML)→z=t(IML) **along e3**, so the ABD through-thickness axis *is* the material e3,
  and the IML mesh sits inside the OML mesh.

## How to read the tables
No single reference is best for every term:
- **Center** is best for the **diagonal** (st15 EA: cen −0.9 % vs OML +1.9 % / IML −3.6 %).
- **OML** is best for the dominant **ext-bend coupling (1,6)** (st15 −9.2 % vs cen −14.5 %)
  — that term is B-driven and the YAML laminate B is defined at the OML.
- **Kirchhoff and RM agree within ~2 %** on these airfoils (no closed-tube pathology).
- Terms at ~1e5 (e.g. st12 ext-twist 1.5e5) are at the solid noise floor — the PNGs drop
  them (`|solid|<1e6`); see the `.py` text output for the full list.

## Reproduce
```
python plot_sections.py     # cross-section figures
python make_table_png.py    # table PNGs
python st_oml_iml_table.py  # verification + text tables
```
(Windows JAX env `C:\conda_envs\opensg_2_0_env\python.exe`.)
