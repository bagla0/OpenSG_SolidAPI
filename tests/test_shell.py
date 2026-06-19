import unittest
import numpy as np
from opensg.mesh.segment import ShellBounMesh
from tests import data_dir, validation_data_dir

# Shell-MSG (Kirchhoff) Timoshenko 6x6 homogenization benchmark on two composite
# wind-turbine-blade airfoil cross-sections (st15, st12). The FEniCS thin-walled
# shell (ShellBounMesh.compute_timo) is validated against:
#   (a) the JAX Hermite-C1 Kirchhoff-shell MSG -- the SAME model class. Its
#       Timoshenko 6x6 is stored in tests/validation_data/test_shell_st{15,12}_jax.txt.
#       The macro gamma_e operator (opensg/utils/shell.py) was made identical to
#       the JAX Ge, bringing the Euler-Bernoulli terms to within ~1%.
#   (b) the 2D-solid VABS / FEniCS-solid (st15) -- a higher-fidelity model. EA/GJ/
#       EI2 within the thin-walled tolerance; EI3 carries the genuine thin-walled
#       over-stiffening (also present in the JAX shell), not an implementation error.
#
# Diagonal Timoshenko order: [EA, GA2, GA3, GJ, EI2, EI3]. The EB terms match the
# JAX shell tightly; the transverse-shear GA2/GA3 are limited by the CG2 +
# interior-penalty vs JAX Hermite-C1 V1s warping discretization (~8%).

ST_MESH = {st: data_dir / "Shell_1DSG" / ("1Dshell_%s.yaml" % st) for st in ("15", "12")}
ST_JAX = {st: validation_data_dir / ("test_shell_st%s_jax.txt" % st) for st in ("15", "12")}
SOLID_15 = dict(EA=1.3082688863e10, GA2=4.5798883250e8, GA3=1.0549929775e8,
                GJ=1.5604378583e8, EI2=1.6630291239e9, EI3=5.1066629243e9)
KEYS = ("EA", "GA2", "GA3", "GJ", "EI2", "EI3")
EB = ("EA", "GJ", "EI2", "EI3")
SHEAR = ("GA2", "GA3")
EB_RTOL, SHEAR_RTOL = 0.02, 0.12


def _diag(M):
    return dict(zip(KEYS, np.diag(np.asarray(M))))


def _shell_timo(st):
    sm = ShellBounMesh(str(ST_MESH[st]))
    ABD, _ = sm.compute_ABD()
    return _diag(sm.compute_timo(ABD)[1])  # Deff_srt (6x6 Timoshenko)


class TestShell(unittest.TestCase):
    def _vs_jax(self, st):
        fe = _shell_timo(st)
        jax = _diag(np.loadtxt(str(ST_JAX[st])))
        for k in EB:
            re = abs(fe[k] - jax[k]) / abs(jax[k])
            print("  st%s %-3s %.4e vs JAX %.4e  %+6.2f%% (tol %.0f%%)"
                  % (st, k, fe[k], jax[k], 100 * (fe[k] - jax[k]) / jax[k], 100 * EB_RTOL))
            assert re < EB_RTOL, "st%s %s rel-err %.4f > %.2f" % (st, k, re, EB_RTOL)
        for k in SHEAR:
            re = abs(fe[k] - jax[k]) / abs(jax[k])
            print("  st%s %-3s %.4e vs JAX %.4e  %+6.2f%% (tol %.0f%%)"
                  % (st, k, fe[k], jax[k], 100 * (fe[k] - jax[k]) / jax[k], 100 * SHEAR_RTOL))
            assert re < SHEAR_RTOL, "st%s %s rel-err %.4f > %.2f" % (st, k, re, SHEAR_RTOL)
        return fe

    def test_shell_homogenization_st15(self):
        """FEniCS Kirchhoff-shell Timoshenko 6x6 vs the JAX shell (.txt) and the
        2D-solid VABS, for the st15 composite section."""
        fe = self._vs_jax("15")
        print("Shell homogenization vs JAX (st15) passed!")
        for k, tol in (("EA", 0.05), ("GJ", 0.05), ("EI2", 0.05), ("EI3", 0.16)):
            re = abs(fe[k] - SOLID_15[k]) / abs(SOLID_15[k])
            print("  st15 %-3s %.4e vs solid %.4e  %+6.2f%% (tol %.0f%%)"
                  % (k, fe[k], SOLID_15[k], 100 * (fe[k] - SOLID_15[k]) / SOLID_15[k], 100 * tol))
            assert re < tol, "st15 %s rel-err %.4f > %.2f" % (k, re, tol)
        print("Shell homogenization vs VABS-solid (st15 EB) passed!")

    def test_shell_homogenization_st12(self):
        """FEniCS Kirchhoff-shell Timoshenko 6x6 vs the JAX shell (.txt) -- st12."""
        self._vs_jax("12")
        print("Shell homogenization vs JAX (st12) passed!")


def run_workflow():
    """Regenerate the JAX reference .txt. Requires the JAX Kirchhoff shell
    (opensg_jax.fe_jax.timoshenko_from_yaml); run in the JAX env, NOT the FEniCS env."""
    from fe_jax import timoshenko_from_yaml
    for st in ("15", "12"):
        T = np.asarray(timoshenko_from_yaml(str(ST_MESH[st]), frac=0.0)[1])
        np.savetxt(str(ST_JAX[st]), T, fmt="%.10e",
                   header="JAX Hermite-C1 Kirchhoff-shell Timoshenko 6x6 (frac=0); diag=[EA,GA2,GA3,GJ,EI2,EI3]")


if __name__ == "__main__":
    unittest.main()
