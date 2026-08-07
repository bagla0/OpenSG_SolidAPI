# Mesh-convergence of the iso / [45/-45] tube benchmark

The isotropic and anisotropic tube benchmark plots were checked for mesh convergence on
**both the shell and the solid** discretizations. They are already at converged resolution,
so the published % error plots are mesh-independent (refining further reproduces them).

## Shell (1D MSG line)
`mesh_convergence.py` sweeps the shell element count NC = 80 → 160 → 320 → 640 and computes
the JAX Kirchhoff + RM Timoshenko 6×6 at h/R=0.06. See `shell_mesh_convergence.png`:
- **Diagonal stiffnesses (EA, GA, GJ, EI):** flat to <0.1 % by the production NC=160.
- **Couplings (ext-twist, shear-bend):** a few % at the coarsest NC=80, converged to <1 % by
  NC=160.
- Earlier full-range test (NC 160→1280) was identical to 5 digits for the shear-bend.

## Solid (2D FEniCSx annulus)
The production solid meshes use NC=200 circumferential × **NR=16 through-thickness = 8 elements
per ply** — well past convergence for a 2-ply laminate (2-4 per ply already converges). A 2×
re-run (NR=24, run via `aniso_tube/sweep_solid.py` in WSL FEniCSx) reproduces the same
reference 6×6; it is not needed to change the benchmark.

## Bottom line
Both meshes are converged at the production resolution; the iso/aniso `*_pcterr.png` plots are
the refined-mesh result. Run `python mesh_convergence.py` to regenerate the convergence figure.
