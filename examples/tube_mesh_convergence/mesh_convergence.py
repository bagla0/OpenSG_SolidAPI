"""Mesh-convergence of the tube Timoshenko 6x6 (iso + [45/-45] aniso) vs the SHELL
1D-line resolution NC.  Shows the stiffnesses (and couplings) are already converged
at the production NC=160, so the iso/aniso benchmark plots are mesh-independent.
(The 2D solid reference uses 16 elements through-thickness = 8 per ply, likewise
converged; a 2x FEniCS solid re-run reproduces the same reference -- see README.)"""
import os, sys, math, yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "aniso_tube"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml
from rm_tube import rm_tube_6x6

DATA = os.path.join(CC, "benchmark_tube", "data")
TMP = os.path.join(HERE, "_tmp"); os.makedirs(TMP, exist_ok=True)
NCS = [80, 160, 320, 640]
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def regen(base_yaml, NC, out):
    d = yaml.safe_load(open(base_yaml))
    n0 = d["nodes"][0]
    R = math.hypot(float(n0[0]), float(n0[1]))                 # OML radius
    step = 2 * math.pi / NC
    d["nodes"] = [[R * math.cos(k * step), R * math.sin(k * step), 0.0] for k in range(NC)]
    d["elements"] = [[k + 1, (k + 1) % NC + 1] for k in range(NC)]
    oris = []
    for k in range(NC):
        tm = (k + 0.5) * step; c, s = math.cos(tm), math.sin(tm)
        oris.append([0.0, 0.0, 1.0, -s, c, 0.0, -c, -s, 0.0])  # e1 axial, e2 tangent, e3 inward
    d["elementOrientations"] = oris
    d["sets"]["element"][0]["labels"] = list(range(1, NC + 1))
    yaml.safe_dump(d, open(out, "w"), default_flow_style=False, sort_keys=True)


def sweep(tag, base_yaml):
    rows_k, rows_r = {}, {}
    for NC in NCS:
        p = os.path.join(TMP, "%s_%d.yaml" % (tag, NC)); regen(base_yaml, NC, p)
        _, K, _ = timoshenko_from_yaml(p, frac=0.5); K = 0.5 * (np.asarray(K) + np.asarray(K).T)
        R6, _ = rm_tube_6x6(p, 0.5); R6 = 0.5 * (R6 + R6.T)
        rows_k[NC] = K; rows_r[NC] = R6
    return rows_k, rows_r


def main():
    cases = [("iso", os.path.join(DATA, "shell_iso_0.06.yaml")),
             ("aniso", os.path.join(DATA, "shell_aniso_0.06.yaml"))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (tag, by) in zip(axes, cases):
        rk, rr = sweep(tag, by)
        ref = rk[NCS[-1]]                                       # finest = reference
        terms = [(LBL[i], i, i) for i in range(6)]
        if tag == "aniso":
            terms += [("ext-tw", 0, 3), ("sh2-b2", 1, 4)]
        for nm, a, b in terms:
            d = [100 * (rk[NC][a, b] - ref[a, b]) / ref[a, b] for NC in NCS]
            ax.plot(NCS, d, "-o", ms=5, label=nm)
        ax.axvline(160, color="k", ls="--", lw=1, label="production NC=160")
        ax.set_xscale("log", base=2); ax.set_xticks(NCS); ax.set_xticklabels(NCS)
        ax.set_xlabel("shell elements NC (log2)")
        ax.set_ylabel("% deviation from finest (NC=640)")
        ax.set_title("%s tube -- JAX Kirchhoff Timoshenko terms vs shell mesh" % tag)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        # convergence table over the MEANINGFUL terms (diagonal + the plotted couplings)
        print("== %s : max |%% dev from finest NC=640| ==" % tag)
        dd = [(i, i) for i in range(6)]
        cc = [(a, b) for nm, a, b in terms if a != b]
        for NC in NCS:
            md = max(abs((rk[NC][i, j] - ref[i, j]) / ref[i, j]) for i, j in dd) * 100
            mc = (max(abs((rk[NC][i, j] - ref[i, j]) / ref[i, j]) for i, j in cc) * 100) if cc else 0.0
            print("   NC=%4d  diagonal max-dev=%.3f%%   coupling max-dev=%.3f%%" % (NC, md, mc))
    fig.suptitle("Shell-mesh convergence: Timoshenko 6x6 is flat by NC=160 (production) -> benchmark plots are mesh-independent",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "shell_mesh_convergence.png")
    fig.savefig(out, dpi=140); print("wrote", out)


if __name__ == "__main__":
    main()
