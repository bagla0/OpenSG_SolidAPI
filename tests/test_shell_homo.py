import unittest
import numpy as np
from opensg.mesh.segment import ShellBounMesh
from tests import data_dir

# Shell-MSG (Kirchhoff) Timoshenko 6x6 homogenization benchmark for two composite
# wind-turbine-blade airfoil cross-sections (st15, st12), comparing the FEniCS
# thin-walled shell against:
#   (a) the JAX Hermite-C1 Kirchhoff-shell MSG -- the SAME model class (tight
#       reference). The macro beam-strain -> plate-strain operator gamma_e
#       (opensg/utils/shell.py) was made identical to the JAX Ge, which brings the
#       Euler-Bernoulli terms (EA, GJ, EI2, EI3) to within ~1%.
#   (b) the 2D-solid VABS / FEniCS-solid (st15) -- a higher-fidelity model. EA/GJ/
#       EI2 are within the thin-walled tolerance; EI3 carries the genuine
#       thin-walled over-stiffening (~14%, also present in the JAX shell), so it is
#       not a shell-implementation error.
#
# Diagonal Timoshenko order: [EA, GA2, GA3, GJ, EI2, EI3].

# JAX-Kirchhoff shell (frac = 0, OML reference):
JAX = {
    "15": dict(EA=1.33262e10, GA2=4.46209e8, GA3=9.62673e7,
               GJ=1.50369e8, EI2=1.67161e9, EI3=5.80882e9),
    "12": dict(EA=1.49884e10, GA2=6.23879e8, GA3=1.33220e8,
               GJ=3.15458e8, EI2=2.93180e9, EI3=9.58438e9),
}
# 2D-solid VABS / FEniCS-solid, st15 (documented higher-fidelity reference):
SOLID_15 = dict(EA=1.3082688863e10, GA2=4.5798883250e8, GA3=1.0549929775e8,
                GJ=1.5604378583e8, EI2=1.6630291239e9, EI3=5.1066629243e9)

# EB terms match the JAX shell tightly; the transverse-shear GA2/GA3 are limited
# by the CG2 + interior-penalty vs JAX Hermite-C1 V1s warping discretization (~8%).
EB_RTOL = 0.02
SHEAR_RTOL = 0.12


def _shell_timo_diag(st):
    sm = ShellBounMesh(str(data_dir / "Shell_1DSG" / ("1Dshell_%s.yaml" % st)))
    ABD, _ = sm.compute_ABD()
    D = np.asarray(sm.compute_timo(ABD)[1])  # Deff_srt (6x6 Timoshenko)
    return dict(EA=D[0, 0], GA2=D[1, 1], GA3=D[2, 2],
                GJ=D[3, 3], EI2=D[4, 4], EI3=D[5, 5])


class TestShellHomo(unittest.TestCase):
    def _check_vs_jax(self, st):
        diag, ref = _shell_timo_diag(st), JAX[st]
        for k in ("EA", "GJ", "EI2", "EI3"):
            re = abs(diag[k] - ref[k]) / abs(ref[k])
            print("  st%s %-3s %.4e vs JAX %.4e  %+6.2f%% (tol %.0f%%)"
                  % (st, k, diag[k], ref[k], 100 * (diag[k] - ref[k]) / ref[k], 100 * EB_RTOL))
            assert re < EB_RTOL, "st%s %s rel-err %.4f > %.2f" % (st, k, re, EB_RTOL)
        for k in ("GA2", "GA3"):
            re = abs(diag[k] - ref[k]) / abs(ref[k])
            print("  st%s %-3s %.4e vs JAX %.4e  %+6.2f%% (tol %.0f%%)"
                  % (st, k, diag[k], ref[k], 100 * (diag[k] - ref[k]) / ref[k], 100 * SHEAR_RTOL))
            assert re < SHEAR_RTOL, "st%s %s rel-err %.4f > %.2f" % (st, k, re, SHEAR_RTOL)

    def test_shell_homo_st15_vs_jax(self):
        """FEniCS Kirchhoff-shell Timoshenko 6x6 vs the JAX shell -- st15."""
        self._check_vs_jax("15")
        print("Shell homogenization vs JAX (st15) passed!")

    def test_shell_homo_st12_vs_jax(self):
        """FEniCS Kirchhoff-shell Timoshenko 6x6 vs the JAX shell -- st12."""
        self._check_vs_jax("12")
        print("Shell homogenization vs JAX (st12) passed!")

    def test_shell_homo_st15_vs_solid(self):
        """st15 EB terms vs the 2D-solid VABS. EA/GJ/EI2 within the thin-walled
        tolerance; EI3 carries the genuine thin-walled over-stiffening (also in the
        JAX shell), checked loosely."""
        diag = _shell_timo_diag("15")
        for k, tol in (("EA", 0.05), ("GJ", 0.05), ("EI2", 0.05), ("EI3", 0.16)):
            re = abs(diag[k] - SOLID_15[k]) / abs(SOLID_15[k])
            print("  st15 %-3s %.4e vs solid %.4e  %+6.2f%% (tol %.0f%%)"
                  % (k, diag[k], SOLID_15[k], 100 * (diag[k] - SOLID_15[k]) / SOLID_15[k], 100 * tol))
            assert re < tol, "st15 %s rel-err %.4f > %.2f" % (k, re, tol)
        print("Shell homogenization vs VABS-solid (st15 EB) passed!")


if __name__ == "__main__":
    unittest.main()
