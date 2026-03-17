import tempfile
import unittest
from pathlib import Path

from dm_labs.hf_utils import ensure_hf_model_card


class HuggingFaceModelCardTests(unittest.TestCase):
    def test_model_card_surfaces_schedule_reweighted_and_uniform_accuracy_views(self):
        eval_summary = {
            "avg_cross_entropy": 1.0,
            "pseudo_perplexity": 2.0,
            "bits_per_masked_token": 3.0,
            "masked_token_accuracy": 0.4,
            "timestep_uniform_avg_cross_entropy": 1.1,
            "timestep_uniform_pseudo_perplexity": 2.1,
            "timestep_uniform_bits_per_masked_token": 3.1,
            "timestep_uniform_masked_token_accuracy": 0.41,
            "schedule_reweighted_avg_cross_entropy": 1.2,
            "schedule_reweighted_pseudo_perplexity": 2.2,
            "schedule_reweighted_bits_per_masked_token": 3.2,
            "schedule_reweighted_masked_token_accuracy": 0.42,
            "grid_uniform_avg_cross_entropy": 1.3,
            "grid_uniform_pseudo_perplexity": 2.3,
            "grid_uniform_bits_per_masked_token": 3.3,
            "grid_uniform_masked_token_accuracy": 0.43,
            "timestep_macro_avg_cross_entropy": 1.4,
            "timestep_macro_pseudo_perplexity": 2.4,
            "timestep_macro_bits_per_masked_token": 3.4,
            "timestep_macro_masked_token_accuracy": 0.44,
            "timestep_auc_avg_cross_entropy": 1.5,
            "timestep_auc_pseudo_perplexity": 2.5,
            "timestep_auc_bits_per_masked_token": 3.5,
            "timestep_auc_masked_token_accuracy": 0.45,
            "metric": "diffusion_pseudo_perplexity",
            "sampled_example_count": 12,
            "masked_tokens": 48,
            "n_batches": 3,
            "timestep_macro_timestep_count": 5,
            "timestep_auc_timestep_count": 5,
            "timestep_auc_fraction_span": 1.0,
            "confidence_intervals": {"pseudo_perplexity": {"p05": 1.8, "p95": 2.2}},
            "schedule_reweighted_confidence_intervals": {"schedule_reweighted_pseudo_perplexity": {"p05": 2.0, "p95": 2.4}},
            "grid_uniform_confidence_intervals": {"grid_uniform_pseudo_perplexity": {"p05": 2.1, "p95": 2.5}},
            "timestep_confidence_intervals": {
                "timestep_macro_pseudo_perplexity": {"p05": 2.2, "p95": 2.6},
                "timestep_auc_pseudo_perplexity": {"p05": 2.3, "p95": 2.7},
            },
            "eval_protocol": {
                "schedule_name": "cosine",
                "T": 64,
                "timestep_grid": [1, 16, 32, 48, 64],
                "timestep_grid_fractions": [0.015625, 0.25, 0.5, 0.75, 1.0],
                "eval_plan_T": 64,
                "normalized_timestep_remapping": False,
                "paired_noise": True,
                "paired_batches": True,
                "sampled_timestep_distribution": "uniform_integer_1_to_eval_plan_T_then_remapped_by_fraction_per_model",
                "schedule_reweighted_aggregation": "inverse_expected_mask_ratio_weighting_over_sampled_masked_tokens",
                "grid_uniform_aggregation": "mean_over_cached_batch_timestep_records",
                "timestep_macro_aggregation": "mean_over_token_weighted_per_timestep_metrics_on_cached_grid",
                "timestep_auc_aggregation": "normalized_trapezoid_integral_over_token_weighted_per_timestep_metrics_on_cached_grid",
                "bootstrap_samples": 1000,
            },
        }

        comparison_summary = {
            "models": [{"tag": "cosine_schedule"}, {"tag": "linear_schedule_baseline"}],
            "delta": {
                "pseudo_perplexity": 0.1,
                "avg_cross_entropy": 0.01,
                "masked_token_accuracy": -0.02,
                "timestep_uniform_pseudo_perplexity": 0.11,
                "timestep_uniform_avg_cross_entropy": 0.011,
                "timestep_uniform_masked_token_accuracy": -0.021,
                "schedule_reweighted_pseudo_perplexity": 0.12,
                "schedule_reweighted_avg_cross_entropy": 0.012,
                "schedule_reweighted_bits_per_masked_token": 0.013,
                "schedule_reweighted_masked_token_accuracy": -0.022,
                "grid_uniform_pseudo_perplexity": 0.13,
                "grid_uniform_avg_cross_entropy": 0.014,
                "grid_uniform_masked_token_accuracy": -0.023,
                "timestep_macro_pseudo_perplexity": 0.14,
                "timestep_macro_avg_cross_entropy": 0.015,
                "timestep_macro_bits_per_masked_token": 0.016,
                "timestep_macro_masked_token_accuracy": -0.024,
                "timestep_auc_pseudo_perplexity": 0.15,
                "timestep_auc_avg_cross_entropy": 0.017,
                "timestep_auc_bits_per_masked_token": 0.018,
                "timestep_auc_masked_token_accuracy": -0.025,
            },
            "winner": {
                "pseudo_perplexity": "cosine_schedule",
                "masked_token_accuracy": "cosine_schedule",
                "timestep_uniform_pseudo_perplexity": "cosine_schedule",
                "timestep_uniform_masked_token_accuracy": "cosine_schedule",
                "schedule_reweighted_pseudo_perplexity": "cosine_schedule",
                "schedule_reweighted_masked_token_accuracy": "cosine_schedule",
                "grid_uniform_pseudo_perplexity": "cosine_schedule",
                "grid_uniform_masked_token_accuracy": "cosine_schedule",
                "timestep_macro_pseudo_perplexity": "cosine_schedule",
                "timestep_macro_masked_token_accuracy": "cosine_schedule",
                "timestep_auc_pseudo_perplexity": "cosine_schedule",
                "timestep_auc_masked_token_accuracy": "cosine_schedule",
            },
            "delta_confidence_intervals": {
                "delta_linear_minus_cosine": {
                    "pseudo_perplexity": {"p05": -0.1, "p95": 0.3, "probability_linear_better": 0.2},
                    "masked_token_accuracy": {"p05": -0.04, "p95": 0.0, "probability_linear_better": 0.1},
                    "timestep_uniform_pseudo_perplexity": {"p05": -0.09, "p95": 0.31, "probability_linear_better": 0.21},
                    "timestep_uniform_masked_token_accuracy": {"p05": -0.041, "p95": 0.001, "probability_linear_better": 0.11},
                    "schedule_reweighted_pseudo_perplexity": {"p05": -0.08, "p95": 0.32, "probability_linear_better": 0.22},
                    "schedule_reweighted_masked_token_accuracy": {"p05": -0.042, "p95": 0.002, "probability_linear_better": 0.12},
                    "grid_uniform_pseudo_perplexity": {"p05": -0.07, "p95": 0.33, "probability_linear_better": 0.23},
                    "grid_uniform_masked_token_accuracy": {"p05": -0.043, "p95": 0.003, "probability_linear_better": 0.13},
                    "timestep_macro_pseudo_perplexity": {"p05": -0.06, "p95": 0.34, "probability_linear_better": 0.24},
                    "timestep_macro_masked_token_accuracy": {"p05": -0.044, "p95": 0.004, "probability_linear_better": 0.14},
                    "timestep_auc_pseudo_perplexity": {"p05": -0.05, "p95": 0.35, "probability_linear_better": 0.25},
                    "timestep_auc_masked_token_accuracy": {"p05": -0.045, "p95": 0.005, "probability_linear_better": 0.15},
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = ensure_hf_model_card(
                tmpdir,
                repo_id="bitlabsdevteam/dm-labs-test",
                eval_summary=eval_summary,
                comparison_summary=comparison_summary,
                overwrite=True,
            )
            content = Path(card_path).read_text(encoding="utf-8")

        self.assertIn("| timestep_uniform_sampled | 1.1 | 2.1 | 3.1 | 0.41 |", content)
        self.assertIn("| schedule_reweighted_sampled | 1.2 | 2.2 | 3.2 | 0.42 |", content)
        self.assertIn("- schedule_reweighted_aggregation: inverse_expected_mask_ratio_weighting_over_sampled_masked_tokens", content)
        self.assertIn("| schedule_reweighted_pseudo_perplexity | 0.12 | cosine_schedule | bootstrap_p05_p95=[-0.08, 0.32], p_linear_better=0.22 |", content)
        self.assertIn("| timestep_uniform_accuracy | -0.021 | cosine_schedule | bootstrap_p05_p95=[-0.041, 0.001], p_linear_better=0.11 |", content)
        self.assertIn("| schedule_reweighted_accuracy | -0.022 | cosine_schedule | bootstrap_p05_p95=[-0.042, 0.002], p_linear_better=0.12 |", content)


if __name__ == "__main__":
    unittest.main()
