import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class CartpolePackageLayoutTest(unittest.TestCase):
    def test_environment_package_and_legacy_wrapper_match_paper_split(self):
        from cartpole.env import CartpoleEnv as PackageCartpoleEnv
        from cartpole_env import CartpoleEnv as LegacyCartpoleEnv

        self.assertIs(LegacyCartpoleEnv, PackageCartpoleEnv)
        self.assertEqual(PackageCartpoleEnv.train_env().cfg.horizon_seconds, 5.0)
        self.assertEqual(PackageCartpoleEnv.train_env().cfg.pole_length, 0.5)
        self.assertEqual(PackageCartpoleEnv.test_env().cfg.horizon_seconds, 300.0)
        self.assertEqual(PackageCartpoleEnv.test_env().cfg.pole_length, 1.0)

    def test_synthesis_legacy_module_alias_preserves_private_patch_points(self):
        import cartpole_synthesis
        from cartpole.psm import synthesis

        self.assertIs(cartpole_synthesis, synthesis)
        with patch("cartpole_synthesis.SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS", 0):
            self.assertEqual(synthesis.SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS, 0)

    def test_direct_opt_legacy_module_alias_preserves_private_patch_points(self):
        import cartpole_direct_opt
        from cartpole.direct_opt import core

        self.assertIs(cartpole_direct_opt, core)
        with patch("cartpole_direct_opt._local_neighbor_candidates", return_value=[]):
            self.assertEqual(core._local_neighbor_candidates(None), [])

    def test_cli_wrappers_keep_main_entry_points(self):
        from cartpole.ppo import train as package_ppo_train
        from cartpole.psm import train as package_psm_train
        from cartpole.direct_opt import train as package_direct_train
        import train_cartpole_ppo
        import train_cartpole_psm
        import train_cartpole_direct_opt

        self.assertTrue(callable(train_cartpole_ppo.main))
        self.assertTrue(callable(train_cartpole_psm.main))
        self.assertTrue(callable(train_cartpole_direct_opt.main))
        self.assertEqual(
            train_cartpole_ppo.PAPER_PPO_TIMESTEPS,
            package_ppo_train.PAPER_PPO_TIMESTEPS,
        )
        self.assertIs(train_cartpole_psm.summarize_student, package_psm_train.summarize_student)
        self.assertIs(
            train_cartpole_direct_opt.DirectOptConfig,
            package_direct_train.DirectOptConfig,
        )


if __name__ == "__main__":
    unittest.main()
