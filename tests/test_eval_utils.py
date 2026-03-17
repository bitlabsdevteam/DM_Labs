import json
import tempfile
import unittest
from pathlib import Path

import torch

from dm_labs.eval_utils import build_eval_plan, compare_schedule_checkpoints, evaluate_diffusion_pseudo_perplexity_from_plan
from dm_labs.modeling import DiffusionLMConfig, DiffusionTransformerLM


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids, timesteps, attention_mask=None):
        batch, seq_len = input_ids.shape
        return self.anchor * torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)


class EvalPlanRemapTests(unittest.TestCase):
    def setUp(self):
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    def test_build_eval_plan_carries_normalized_timestep_fractions(self):
        plan = build_eval_plan([self.batch], T=10, n_batches=1, timestep_grid=[5, 10], seed=0)
        timestep_plans = plan["batches"][0]["timestep_plans"]
        self.assertIn("u", timestep_plans[0])
        self.assertEqual(timestep_plans[1]["u"].tolist(), [0.5])
        self.assertEqual(timestep_plans[2]["u"].tolist(), [1.0])

    def test_evaluation_remaps_shared_plan_timesteps_by_fraction(self):
        plan = build_eval_plan([self.batch], T=10, n_batches=1, timestep_grid=[5, 10], seed=0)
        model = ConstantLogitModel(vocab_size=8)

        def corruption_fn(input_ids, attention_mask, t, mask_token_id, T, excluded_token_ids=None, rand=None, generator=None):
            labels = input_ids.clone()
            mask_positions = torch.ones_like(input_ids, dtype=torch.bool)
            noisy = torch.full_like(input_ids, mask_token_id)
            return noisy, labels, mask_positions

        result = evaluate_diffusion_pseudo_perplexity_from_plan(
            model=model,
            eval_plan=plan,
            corruption_fn=corruption_fn,
            mask_token_id=0,
            T=20,
            schedule_name="cosine",
            bootstrap_samples=8,
        )
        metrics = result["timestep_metrics"]
        self.assertEqual([row["source_plan_timestep"] for row in metrics], [5, 10])
        self.assertEqual([row["timestep"] for row in metrics], [10, 20])
        self.assertEqual([row["timestep_fraction"] for row in metrics], [0.5, 1.0])
        self.assertTrue(result["eval_protocol"]["normalized_timestep_remapping"])
        self.assertIn("schedule_reweighted_avg_cross_entropy", result)
        self.assertIn("schedule_reweighted_pseudo_perplexity", result)
        self.assertIn("schedule_reweighted_aggregation", result["eval_protocol"])

    def test_timestep_auc_matches_constant_metric_surface(self):
        plan = {
            "n_batches": 1,
            "seed": 0,
            "T": 10,
            "timestep_grid": [2, 5, 10],
            "batches": [
                {
                    "input_ids": self.batch["input_ids"],
                    "attention_mask": self.batch["attention_mask"],
                    "active_tokens": 4,
                    "timestep_plans": [
                        {"kind": "grid", "t": torch.tensor([2]), "u": torch.tensor([0.2]), "rand": torch.zeros((1, 4))},
                        {"kind": "grid", "t": torch.tensor([5]), "u": torch.tensor([0.5]), "rand": torch.zeros((1, 4))},
                        {"kind": "grid", "t": torch.tensor([10]), "u": torch.tensor([1.0]), "rand": torch.zeros((1, 4))},
                    ],
                }
            ],
        }
        model = ConstantLogitModel(vocab_size=8)

        def corruption_fn(input_ids, attention_mask, t, mask_token_id, T, excluded_token_ids=None, rand=None, generator=None):
            labels = input_ids.clone()
            mask_positions = torch.ones_like(input_ids, dtype=torch.bool)
            noisy = torch.full_like(input_ids, mask_token_id)
            return noisy, labels, mask_positions

        result = evaluate_diffusion_pseudo_perplexity_from_plan(
            model=model,
            eval_plan=plan,
            corruption_fn=corruption_fn,
            mask_token_id=0,
            T=10,
            schedule_name="cosine",
            bootstrap_samples=8,
        )
        expected_ce = torch.log(torch.tensor(8.0)).item()
        self.assertAlmostEqual(result["timestep_auc_avg_cross_entropy"], expected_ce, places=6)
        self.assertAlmostEqual(result["timestep_auc_pseudo_perplexity"], 8.0, places=6)
        self.assertAlmostEqual(result["timestep_auc_masked_token_accuracy"], 0.125, places=6)
        self.assertAlmostEqual(result["timestep_auc_fraction_span"], 0.8, places=6)
        self.assertEqual(result["timestep_auc_timestep_count"], 3)
        self.assertAlmostEqual(result["schedule_reweighted_avg_cross_entropy"], expected_ce, places=6)
        self.assertAlmostEqual(result["schedule_reweighted_pseudo_perplexity"], 8.0, places=6)
        self.assertAlmostEqual(result["schedule_reweighted_masked_token_accuracy"], 0.125, places=6)
        self.assertAlmostEqual(result["timestep_uniform_masked_token_accuracy"], 0.125, places=6)
        timestep_uniform_ci = result["timestep_uniform_confidence_intervals"]
        self.assertEqual(timestep_uniform_ci["n_examples"], 3)
        self.assertIn("timestep_uniform_pseudo_perplexity", timestep_uniform_ci)
        self.assertIn("timestep_uniform_masked_token_accuracy", timestep_uniform_ci)
        reweighted_ci = result["schedule_reweighted_confidence_intervals"]
        self.assertIn("schedule_reweighted_pseudo_perplexity", reweighted_ci)
        self.assertIn("schedule_reweighted_masked_token_accuracy", reweighted_ci)
        timestep_ci = result["timestep_confidence_intervals"]
        self.assertEqual(timestep_ci["n_timesteps"], 3)
        self.assertIn("timestep_macro_pseudo_perplexity", timestep_ci)
        self.assertIn("timestep_auc_pseudo_perplexity", timestep_ci)


class ScheduleComparisonRemapTests(unittest.TestCase):
    def _write_checkpoint(self, root: Path, diffusion_steps: int) -> Path:
        cfg = DiffusionLMConfig(
            vocab_size=16,
            seq_len=4,
            d_model=8,
            n_layers=1,
            n_heads=1,
            d_ff=16,
            dropout=0.0,
            diffusion_steps=diffusion_steps,
        )
        model = DiffusionTransformerLM(cfg)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(cfg.__dict__), encoding="utf-8")
        torch.save(model.state_dict(), root / "model.pt")
        return root

    def test_schedule_comparison_reports_shared_source_timestep_keys(self):
        dataloader = [{
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cosine_dir = self._write_checkpoint(tmp / "cosine", diffusion_steps=10)
            linear_dir = self._write_checkpoint(tmp / "linear", diffusion_steps=20)
            comparison = compare_schedule_checkpoints(
                cosine_dir=cosine_dir,
                linear_dir=linear_dir,
                device=torch.device("cpu"),
                config_cls=DiffusionLMConfig,
                model_cls=DiffusionTransformerLM,
                dataloader=dataloader,
                mask_token_id=0,
                excluded_token_ids=None,
                n_batches=1,
                timestep_grid=[5, 10],
                seed=0,
                bootstrap_samples=8,
            )

        self.assertTrue(comparison["comparison_protocol"]["normalized_timestep_remapping"])
        deltas = comparison["timestep_deltas"]
        self.assertEqual([row["source_plan_timestep"] for row in deltas], [5, 10])
        self.assertEqual([row["cosine_timestep"] for row in deltas], [5, 10])
        self.assertEqual([row["linear_timestep"] for row in deltas], [10, 20])
        self.assertIn("timestep_auc_pseudo_perplexity", comparison["delta"])
        self.assertIn("timestep_auc_pseudo_perplexity", comparison["winner"])
        self.assertIn("schedule_reweighted_pseudo_perplexity", comparison["delta"])
        self.assertIn("schedule_reweighted_masked_token_accuracy", comparison["winner"])
        delta_ci = comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"]
        self.assertIn("timestep_auc_pseudo_perplexity", delta_ci)
        self.assertIn("timestep_auc_masked_token_accuracy", delta_ci)
        self.assertIn("schedule_reweighted_pseudo_perplexity", delta_ci)
        self.assertIn("schedule_reweighted_masked_token_accuracy", delta_ci)


if __name__ == "__main__":
    unittest.main()
