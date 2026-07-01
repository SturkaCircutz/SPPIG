import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_direct_opt.py")

from cartpole_direct_opt import DirectOptConfig, direct_opt_metrics, run_cartpole_direct_opt  # noqa: E402


class CartpoleDirectOptTest(unittest.TestCase):
    def test_direct_opt_returns_policy_and_provenance(self):
        result = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=0,
                num_train_states=2,
                random_candidates=4,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
            )
        )
        metrics = direct_opt_metrics(result)

        self.assertEqual(result.searched_candidates, 160)
        self.assertIn("mode=1 if", metrics["policy_description"])
        self.assertEqual(metrics["algorithm_provenance"]["paper_baseline"], "Direct-Opt")
        self.assertTrue(metrics["algorithm_provenance"]["not_paper_scale"])
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("train", metrics)
        self.assertIn("test", metrics)

    def test_direct_opt_cli_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "direct_opt_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--metrics-output",
                    metrics_path,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertEqual(metrics["config"]["quick"], True)
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["algorithm_provenance"]["baseline"], "direct_opt")
        self.assertIn("best_candidate", metrics)


if __name__ == "__main__":
    unittest.main()
