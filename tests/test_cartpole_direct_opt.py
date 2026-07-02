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

        diagnostics = metrics["search_diagnostics"]
        expected_evaluations = (
            diagnostics["grid_candidates"]
            + diagnostics["random_candidates"]
            + diagnostics["batch_refinement_candidates"]
            + diagnostics["batch_seed_evaluations"]
            + diagnostics["batch_local_evaluations"]
            + diagnostics["restart_evaluations"]
        )
        self.assertEqual(result.searched_candidates, expected_evaluations)
        self.assertIn("mode=1 if", metrics["policy_description"])
        self.assertEqual(metrics["algorithm_provenance"]["paper_baseline"], "Direct-Opt")
        self.assertTrue(metrics["algorithm_provenance"]["not_paper_scale"])
        self.assertEqual(metrics["algorithm_provenance"]["paper_batch_size"], 10)
        self.assertEqual(metrics["algorithm_provenance"]["paper_parallel_threads"], 10)
        self.assertEqual(metrics["algorithm_provenance"]["paper_time_limit_seconds"], 7200)
        self.assertEqual(metrics["algorithm_provenance"]["local_parallel_threads"], 1)
        self.assertIn("bounded batch/restart", metrics["algorithm_provenance"]["limitations"])
        self.assertEqual(diagnostics["grid_candidates"], 156)
        self.assertEqual(diagnostics["random_candidates"], 4)
        self.assertEqual(diagnostics["batch_count"], 1)
        self.assertEqual(diagnostics["batch_rounds"], 1)
        self.assertEqual(diagnostics["batch_refinement_candidates"], 1)
        self.assertEqual(diagnostics["batch_seed_evaluations"], 1)
        self.assertGreater(diagnostics["batch_local_evaluations"], 0)
        self.assertGreaterEqual(diagnostics["restart_evaluations"], 0)
        self.assertEqual(metrics["best_candidate"]["source"], result.candidate.source)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("train", metrics)
        self.assertIn("test", metrics)

    def test_direct_opt_can_disable_batch_refinement_for_grid_random_diagnostic(self):
        result = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=0,
                num_train_states=2,
                random_candidates=4,
                batch_refinement_rounds=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
            )
        )

        self.assertEqual(result.searched_candidates, 160)
        self.assertEqual(result.search_diagnostics["batch_refinement_candidates"], 0)
        self.assertEqual(result.search_diagnostics["batch_seed_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["batch_local_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["restart_evaluations"], 0)

    def test_direct_opt_batch_refinement_preserves_full_train_best_so_far(self):
        base_cfg = DirectOptConfig(
            seed=1,
            num_train_states=3,
            random_candidates=4,
            batch_size=1,
            batch_refinement_rounds=0,
            eval_rollouts=1,
            test_max_steps=20,
            quick=True,
        )
        refined_cfg = DirectOptConfig(
            seed=1,
            num_train_states=3,
            random_candidates=4,
            batch_size=1,
            batch_refinement_rounds=2,
            local_refinement_steps=1,
            restart_candidates_on_stall=1,
            eval_rollouts=1,
            test_max_steps=20,
            quick=True,
        )

        base = run_cartpole_direct_opt(base_cfg)
        refined = run_cartpole_direct_opt(refined_cfg)

        self.assertGreaterEqual(
            refined.candidate.train_reward_mean,
            base.candidate.train_reward_mean,
        )
        self.assertGreater(refined.search_diagnostics["batch_refinement_candidates"], 0)

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
        self.assertEqual(metrics["config"]["batch_size"], 2)
        self.assertEqual(metrics["config"]["batch_refinement_rounds"], 1)
        self.assertEqual(metrics["config"]["local_refinement_steps"], 1)
        self.assertEqual(metrics["config"]["restart_candidates_on_stall"], 1)
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["algorithm_provenance"]["baseline"], "direct_opt")
        self.assertIn("search_diagnostics", metrics)
        self.assertIn("best_candidate", metrics)

    def test_direct_opt_cli_quick_honors_disabled_batch_refinement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "direct_opt_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--batch-refinement-rounds",
                    "0",
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
        self.assertEqual(metrics["config"]["batch_refinement_rounds"], 0)
        self.assertEqual(metrics["search_diagnostics"]["batch_refinement_candidates"], 0)
        self.assertEqual(metrics["search_diagnostics"]["batch_seed_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
