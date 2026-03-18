import tempfile
import unittest
from pathlib import Path

from dm_labs.hf_utils import build_eval_view_rows, build_schedule_comparison_rows, ensure_hf_model_card, validate_hf_export_bundle, write_eval_plan, write_hf_export_bundle


class HuggingFaceModelCardTests(unittest.TestCase):
    def test_model_card_surfaces_schedule_reweighted_and_uniform_accuracy_views(self):
        eval_summary = {
            "vocab_size": 16,
            "uniform_random_avg_cross_entropy": 2.772588722239781,
            "uniform_random_pseudo_perplexity": 16.0,
            "uniform_random_bits_per_masked_token": 4.0,
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
            "schedule_reweighted_nonzero_examples": 12,
            "schedule_reweighted_estimated_eligible_token_count": 60.0,
            "schedule_reweighted_effective_sample_size": 10.5,
            "schedule_reweighted_effective_sample_size_fraction": 0.875,
            "timestep_macro_timestep_count": 5,
            "timestep_auc_timestep_count": 5,
            "timestep_auc_fraction_span": 1.0,
            "confidence_intervals": {
                "avg_cross_entropy": {"p05": 0.9, "p95": 1.1},
                "pseudo_perplexity": {"p05": 1.8, "p95": 2.2},
                "masked_token_accuracy": {"p05": 0.3, "p95": 0.5},
            },
            "timestep_uniform_confidence_intervals": {
                "timestep_uniform_avg_cross_entropy": {"p05": 1.0, "p95": 1.2},
                "timestep_uniform_pseudo_perplexity": {"p05": 1.9, "p95": 2.3},
                "timestep_uniform_masked_token_accuracy": {"p05": 0.31, "p95": 0.51},
            },
            "schedule_reweighted_confidence_intervals": {
                "schedule_reweighted_avg_cross_entropy": {"p05": 1.1, "p95": 1.3},
                "schedule_reweighted_pseudo_perplexity": {"p05": 2.0, "p95": 2.4},
                "schedule_reweighted_masked_token_accuracy": {"p05": 0.32, "p95": 0.52},
            },
            "grid_uniform_confidence_intervals": {
                "grid_uniform_avg_cross_entropy": {"p05": 1.2, "p95": 1.4},
                "grid_uniform_pseudo_perplexity": {"p05": 2.1, "p95": 2.5},
                "grid_uniform_masked_token_accuracy": {"p05": 0.33, "p95": 0.53},
            },
            "timestep_confidence_intervals": {
                "timestep_macro_avg_cross_entropy": {"p05": 1.3, "p95": 1.5},
                "timestep_auc_avg_cross_entropy": {"p05": 1.4, "p95": 1.6},
                "timestep_macro_pseudo_perplexity": {"p05": 2.2, "p95": 2.6},
                "timestep_auc_pseudo_perplexity": {"p05": 2.3, "p95": 2.7},
                "timestep_macro_masked_token_accuracy": {"p05": 0.34, "p95": 0.54},
                "timestep_auc_masked_token_accuracy": {"p05": 0.35, "p95": 0.55},
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
            "winner_confidence": {
                "pseudo_perplexity": {"winner_probability": 0.8, "ci_excludes_zero": False, "practically_tied": True},
                "bits_per_masked_token": {"winner_probability": 0.8, "ci_excludes_zero": False, "practically_tied": True},
                "sampled_bits_saved_vs_uniform": {"winner_probability": 0.8, "ci_excludes_zero": False, "practically_tied": True},
                "sampled_denoising_skill": {"winner_probability": 0.8, "ci_excludes_zero": False, "practically_tied": True},
                "masked_token_accuracy": {"winner_probability": 0.9, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_uniform_pseudo_perplexity": {"winner_probability": 0.79, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_uniform_bits_per_masked_token": {"winner_probability": 0.79, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_uniform_bits_saved_vs_uniform": {"winner_probability": 0.79, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_uniform_denoising_skill": {"winner_probability": 0.79, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_uniform_masked_token_accuracy": {"winner_probability": 0.89, "ci_excludes_zero": False, "practically_tied": True},
                "schedule_reweighted_pseudo_perplexity": {"winner_probability": 0.78, "ci_excludes_zero": False, "practically_tied": True},
                "schedule_reweighted_bits_per_masked_token": {"winner_probability": 0.78, "ci_excludes_zero": False, "practically_tied": True},
                "schedule_reweighted_bits_saved_vs_uniform": {"winner_probability": 0.78, "ci_excludes_zero": False, "practically_tied": True},
                "schedule_reweighted_denoising_skill": {"winner_probability": 0.78, "ci_excludes_zero": False, "practically_tied": True},
                "schedule_reweighted_masked_token_accuracy": {"winner_probability": 0.88, "ci_excludes_zero": False, "practically_tied": True},
                "grid_uniform_pseudo_perplexity": {"winner_probability": 0.77, "ci_excludes_zero": False, "practically_tied": True},
                "grid_uniform_bits_per_masked_token": {"winner_probability": 0.77, "ci_excludes_zero": False, "practically_tied": True},
                "grid_uniform_bits_saved_vs_uniform": {"winner_probability": 0.77, "ci_excludes_zero": False, "practically_tied": True},
                "grid_uniform_denoising_skill": {"winner_probability": 0.77, "ci_excludes_zero": False, "practically_tied": True},
                "grid_uniform_masked_token_accuracy": {"winner_probability": 0.87, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_macro_pseudo_perplexity": {"winner_probability": 0.76, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_macro_bits_per_masked_token": {"winner_probability": 0.76, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_macro_bits_saved_vs_uniform": {"winner_probability": 0.76, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_macro_denoising_skill": {"winner_probability": 0.76, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_macro_masked_token_accuracy": {"winner_probability": 0.86, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_auc_pseudo_perplexity": {"winner_probability": 0.75, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_auc_bits_per_masked_token": {"winner_probability": 0.75, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_auc_bits_saved_vs_uniform": {"winner_probability": 0.75, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_auc_denoising_skill": {"winner_probability": 0.75, "ci_excludes_zero": False, "practically_tied": True},
                "timestep_auc_masked_token_accuracy": {"winner_probability": 0.85, "ci_excludes_zero": False, "practically_tied": True},
            },
            "delta": {
                "pseudo_perplexity": 0.1,
                "avg_cross_entropy": 0.01,
                "bits_per_masked_token": 0.014426950408889635,
                "masked_token_accuracy": -0.02,
                "sampled_bits_saved_vs_uniform": -0.014426950408889635,
                "sampled_denoising_skill": -0.0036067376022224087,
                "timestep_uniform_pseudo_perplexity": 0.11,
                "timestep_uniform_avg_cross_entropy": 0.011,
                "timestep_uniform_bits_per_masked_token": 0.015869645449778603,
                "timestep_uniform_masked_token_accuracy": -0.021,
                "timestep_uniform_bits_saved_vs_uniform": -0.015869645449778603,
                "timestep_uniform_denoising_skill": -0.003967411362444651,
                "schedule_reweighted_pseudo_perplexity": 0.12,
                "schedule_reweighted_avg_cross_entropy": 0.012,
                "schedule_reweighted_bits_per_masked_token": 0.013,
                "schedule_reweighted_masked_token_accuracy": -0.022,
                "schedule_reweighted_bits_saved_vs_uniform": -0.013,
                "schedule_reweighted_denoising_skill": -0.004328085122666838,
                "grid_uniform_pseudo_perplexity": 0.13,
                "grid_uniform_avg_cross_entropy": 0.014,
                "grid_uniform_bits_per_masked_token": 0.02019773057244548,
                "grid_uniform_masked_token_accuracy": -0.023,
                "grid_uniform_bits_saved_vs_uniform": -0.02019773057244548,
                "grid_uniform_denoising_skill": -0.005049432643111315,
                "timestep_macro_pseudo_perplexity": 0.14,
                "timestep_macro_avg_cross_entropy": 0.015,
                "timestep_macro_bits_per_masked_token": 0.016,
                "timestep_macro_masked_token_accuracy": -0.024,
                "timestep_macro_bits_saved_vs_uniform": -0.016,
                "timestep_macro_denoising_skill": -0.0054101064033335014,
                "timestep_auc_pseudo_perplexity": 0.15,
                "timestep_auc_avg_cross_entropy": 0.017,
                "timestep_auc_bits_per_masked_token": 0.018,
                "timestep_auc_masked_token_accuracy": -0.025,
                "timestep_auc_bits_saved_vs_uniform": -0.018,
                "timestep_auc_denoising_skill": -0.005770780163555688,
            },
            "winner": {
                "pseudo_perplexity": "cosine_schedule",
                "bits_per_masked_token": "cosine_schedule",
                "masked_token_accuracy": "cosine_schedule",
                "sampled_bits_saved_vs_uniform": "cosine_schedule",
                "sampled_denoising_skill": "cosine_schedule",
                "timestep_uniform_pseudo_perplexity": "cosine_schedule",
                "timestep_uniform_bits_per_masked_token": "cosine_schedule",
                "timestep_uniform_masked_token_accuracy": "cosine_schedule",
                "timestep_uniform_bits_saved_vs_uniform": "cosine_schedule",
                "timestep_uniform_denoising_skill": "cosine_schedule",
                "schedule_reweighted_pseudo_perplexity": "cosine_schedule",
                "schedule_reweighted_bits_per_masked_token": "cosine_schedule",
                "schedule_reweighted_masked_token_accuracy": "cosine_schedule",
                "schedule_reweighted_bits_saved_vs_uniform": "cosine_schedule",
                "schedule_reweighted_denoising_skill": "cosine_schedule",
                "grid_uniform_pseudo_perplexity": "cosine_schedule",
                "grid_uniform_bits_per_masked_token": "cosine_schedule",
                "grid_uniform_masked_token_accuracy": "cosine_schedule",
                "grid_uniform_bits_saved_vs_uniform": "cosine_schedule",
                "grid_uniform_denoising_skill": "cosine_schedule",
                "timestep_macro_pseudo_perplexity": "cosine_schedule",
                "timestep_macro_bits_per_masked_token": "cosine_schedule",
                "timestep_macro_masked_token_accuracy": "cosine_schedule",
                "timestep_macro_bits_saved_vs_uniform": "cosine_schedule",
                "timestep_macro_denoising_skill": "cosine_schedule",
                "timestep_auc_pseudo_perplexity": "cosine_schedule",
                "timestep_auc_bits_per_masked_token": "cosine_schedule",
                "timestep_auc_masked_token_accuracy": "cosine_schedule",
                "timestep_auc_bits_saved_vs_uniform": "cosine_schedule",
                "timestep_auc_denoising_skill": "cosine_schedule",
            },
            "delta_confidence_intervals": {
                "delta_linear_minus_cosine": {
                    "pseudo_perplexity": {"p05": -0.1, "p95": 0.3, "probability_linear_better": 0.2},
                    "bits_per_masked_token": {"p05": -0.02, "p95": 0.05, "probability_linear_better": 0.2},
                    "sampled_bits_saved_vs_uniform": {"p05": -0.05, "p95": 0.02, "probability_linear_better": 0.8},
                    "sampled_denoising_skill": {"p05": -0.018, "p95": 0.007, "probability_linear_better": 0.8},
                    "masked_token_accuracy": {"p05": -0.04, "p95": 0.0, "probability_linear_better": 0.1},
                    "timestep_uniform_pseudo_perplexity": {"p05": -0.09, "p95": 0.31, "probability_linear_better": 0.21},
                    "timestep_uniform_bits_per_masked_token": {"p05": -0.021, "p95": 0.051, "probability_linear_better": 0.21},
                    "timestep_uniform_bits_saved_vs_uniform": {"p05": -0.051, "p95": 0.021, "probability_linear_better": 0.79},
                    "timestep_uniform_denoising_skill": {"p05": -0.019, "p95": 0.008, "probability_linear_better": 0.79},
                    "timestep_uniform_masked_token_accuracy": {"p05": -0.041, "p95": 0.001, "probability_linear_better": 0.11},
                    "schedule_reweighted_pseudo_perplexity": {"p05": -0.08, "p95": 0.32, "probability_linear_better": 0.22},
                    "schedule_reweighted_bits_per_masked_token": {"p05": -0.022, "p95": 0.052, "probability_linear_better": 0.22},
                    "schedule_reweighted_bits_saved_vs_uniform": {"p05": -0.052, "p95": 0.022, "probability_linear_better": 0.78},
                    "schedule_reweighted_denoising_skill": {"p05": -0.02, "p95": 0.009, "probability_linear_better": 0.78},
                    "schedule_reweighted_masked_token_accuracy": {"p05": -0.042, "p95": 0.002, "probability_linear_better": 0.12},
                    "grid_uniform_pseudo_perplexity": {"p05": -0.07, "p95": 0.33, "probability_linear_better": 0.23},
                    "grid_uniform_bits_per_masked_token": {"p05": -0.023, "p95": 0.053, "probability_linear_better": 0.23},
                    "grid_uniform_bits_saved_vs_uniform": {"p05": -0.053, "p95": 0.023, "probability_linear_better": 0.77},
                    "grid_uniform_denoising_skill": {"p05": -0.021, "p95": 0.01, "probability_linear_better": 0.77},
                    "grid_uniform_masked_token_accuracy": {"p05": -0.043, "p95": 0.003, "probability_linear_better": 0.13},
                    "timestep_macro_pseudo_perplexity": {"p05": -0.06, "p95": 0.34, "probability_linear_better": 0.24},
                    "timestep_macro_bits_per_masked_token": {"p05": -0.024, "p95": 0.054, "probability_linear_better": 0.24},
                    "timestep_macro_bits_saved_vs_uniform": {"p05": -0.054, "p95": 0.024, "probability_linear_better": 0.76},
                    "timestep_macro_denoising_skill": {"p05": -0.022, "p95": 0.011, "probability_linear_better": 0.76},
                    "timestep_macro_masked_token_accuracy": {"p05": -0.044, "p95": 0.004, "probability_linear_better": 0.14},
                    "timestep_auc_pseudo_perplexity": {"p05": -0.05, "p95": 0.35, "probability_linear_better": 0.25},
                    "timestep_auc_bits_per_masked_token": {"p05": -0.025, "p95": 0.055, "probability_linear_better": 0.25},
                    "timestep_auc_bits_saved_vs_uniform": {"p05": -0.055, "p95": 0.025, "probability_linear_better": 0.75},
                    "timestep_auc_denoising_skill": {"p05": -0.023, "p95": 0.012, "probability_linear_better": 0.75},
                    "timestep_auc_masked_token_accuracy": {"p05": -0.045, "p95": 0.005, "probability_linear_better": 0.15},
                }
            },
        }

        comparison_summary["models"] = [
            {
                "tag": "cosine_schedule",
                "pseudo_perplexity": 2.0,
                "bits_per_masked_token": 3.0,
                "masked_token_accuracy": 0.4,
                "timestep_uniform_pseudo_perplexity": 2.1,
                "timestep_uniform_bits_per_masked_token": 3.1,
                "timestep_uniform_masked_token_accuracy": 0.41,
                "schedule_reweighted_pseudo_perplexity": 2.2,
                "schedule_reweighted_bits_per_masked_token": 3.2,
                "schedule_reweighted_masked_token_accuracy": 0.42,
                "grid_uniform_pseudo_perplexity": 2.3,
                "grid_uniform_bits_per_masked_token": 3.3,
                "grid_uniform_masked_token_accuracy": 0.43,
                "timestep_macro_pseudo_perplexity": 2.4,
                "timestep_macro_bits_per_masked_token": 3.4,
                "timestep_macro_masked_token_accuracy": 0.44,
                "timestep_auc_pseudo_perplexity": 2.5,
                "timestep_auc_bits_per_masked_token": 3.5,
                "timestep_auc_masked_token_accuracy": 0.45,
            },
            {
                "tag": "linear_schedule_baseline",
                "pseudo_perplexity": 2.1,
                "bits_per_masked_token": 3.0144269504088897,
                "masked_token_accuracy": 0.38,
                "timestep_uniform_pseudo_perplexity": 2.21,
                "timestep_uniform_bits_per_masked_token": 3.1158696454497787,
                "timestep_uniform_masked_token_accuracy": 0.389,
                "schedule_reweighted_pseudo_perplexity": 2.32,
                "schedule_reweighted_bits_per_masked_token": 3.213,
                "schedule_reweighted_masked_token_accuracy": 0.398,
                "grid_uniform_pseudo_perplexity": 2.43,
                "grid_uniform_bits_per_masked_token": 3.3201977305724456,
                "grid_uniform_masked_token_accuracy": 0.407,
                "timestep_macro_pseudo_perplexity": 2.54,
                "timestep_macro_bits_per_masked_token": 3.416,
                "timestep_macro_masked_token_accuracy": 0.416,
                "timestep_auc_pseudo_perplexity": 2.65,
                "timestep_auc_bits_per_masked_token": 3.518,
                "timestep_auc_masked_token_accuracy": 0.425,
            },
        ]
        comparison_summary["calibration"] = {
            "cosine_schedule": {
                "sampled": {"bits_saved_vs_uniform": 1.0, "denoising_skill": 0.6393262397777592},
                "timestep_uniform": {"bits_saved_vs_uniform": 0.9, "denoising_skill": 0.603258863755535},
                "schedule_reweighted": {"bits_saved_vs_uniform": 0.8, "denoising_skill": 0.567191487733311},
                "grid_uniform": {"bits_saved_vs_uniform": 0.7, "denoising_skill": 0.5311241117110867},
                "timestep_macro": {"bits_saved_vs_uniform": 0.6, "denoising_skill": 0.49505673568886246},
                "timestep_auc": {"bits_saved_vs_uniform": 0.5, "denoising_skill": 0.45898935966663814},
            },
            "linear_schedule_baseline": {
                "sampled": {"bits_saved_vs_uniform": 0.9855730495911104, "denoising_skill": 0.6357195021755368},
                "timestep_uniform": {"bits_saved_vs_uniform": 0.8841303545502214, "denoising_skill": 0.5992914523930904},
                "schedule_reweighted": {"bits_saved_vs_uniform": 0.787, "denoising_skill": 0.5628634026106441},
                "grid_uniform": {"bits_saved_vs_uniform": 0.6798022694275545, "denoising_skill": 0.5260746790679754},
                "timestep_macro": {"bits_saved_vs_uniform": 0.584, "denoising_skill": 0.489646629285529},
                "timestep_auc": {"bits_saved_vs_uniform": 0.482, "denoising_skill": 0.45321857950308244},
            },
        }

        eval_rows = build_eval_view_rows(eval_summary)
        self.assertEqual(eval_rows[0]["view"], "token_weighted_sampled")
        self.assertEqual(eval_rows[0]["uniform_random_pseudo_perplexity"], 16.0)
        self.assertAlmostEqual(eval_rows[0]["bits_saved_vs_uniform"], 1.0, places=6)
        self.assertAlmostEqual(eval_rows[0]["denoising_skill"], 1.0 - (1.0 / 2.772588722239781), places=6)
        self.assertEqual(eval_rows[0]["pseudo_perplexity_ci_p05"], 1.8)
        self.assertEqual(eval_rows[0]["masked_token_accuracy_ci_p95"], 0.5)
        self.assertAlmostEqual(eval_rows[0]["bits_saved_vs_uniform_ci_p05"], 4.0 - (1.1 / 0.6931471805599453), places=6)
        self.assertAlmostEqual(eval_rows[0]["bits_saved_vs_uniform_ci_p95"], 4.0 - (0.9 / 0.6931471805599453), places=6)
        self.assertAlmostEqual(eval_rows[0]["denoising_skill_ci_p05"], 1.0 - (1.1 / 2.772588722239781), places=6)
        self.assertAlmostEqual(eval_rows[0]["denoising_skill_ci_p95"], 1.0 - (0.9 / 2.772588722239781), places=6)
        self.assertEqual(eval_rows[1]["pseudo_perplexity_ci_p05"], 1.9)
        self.assertEqual(eval_rows[1]["masked_token_accuracy_ci_p95"], 0.51)
        self.assertEqual(eval_rows[2]["aggregation"], "inverse-expected-mask-ratio weighting over sampled masked tokens")
        self.assertEqual(eval_rows[-1]["masked_token_accuracy_ci_p05"], 0.35)

        comparison_rows = build_schedule_comparison_rows(comparison_summary)
        self.assertEqual(comparison_rows[0]["metric_view"], "sampled_pseudo_perplexity")
        self.assertEqual(comparison_rows[0]["better_direction"], "lower")
        self.assertEqual(comparison_rows[1]["metric_view"], "sampled_bits_per_masked_token")
        self.assertEqual(comparison_rows[1]["cosine_value"], 3.0)
        self.assertEqual(comparison_rows[1]["linear_value"], 3.0144269504088897)
        self.assertEqual(comparison_rows[1]["bootstrap_p95"], 0.05)
        self.assertEqual(comparison_rows[1]["winner_probability"], 0.8)
        self.assertFalse(comparison_rows[1]["ci_excludes_zero"])
        self.assertTrue(comparison_rows[1]["practically_tied"])
        comparison_rows_by_view = {row["metric_view"]: row for row in comparison_rows}
        self.assertEqual(comparison_rows_by_view["timestep_auc_bits_saved_vs_uniform"]["winner"], "cosine_schedule")
        self.assertEqual(comparison_rows_by_view["timestep_auc_bits_saved_vs_uniform"]["bootstrap_p95"], 0.025)
        self.assertEqual(comparison_rows_by_view["timestep_auc_bits_saved_vs_uniform"]["probability_linear_better"], 0.75)
        self.assertEqual(comparison_rows[-1]["bootstrap_p95"], 0.005)
        self.assertEqual(comparison_rows[-1]["probability_linear_better"], 0.15)

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_plan_path = write_eval_plan(tmpdir, eval_plan={"n_batches": 3, "T": 64, "timestep_grid": [1, 16, 32, 48, 64], "seed": 7, "batches": []})
            self.assertTrue(Path(eval_plan_path).exists())
            bundle = write_hf_export_bundle(
                tmpdir,
                repo_id="bitlabsdevteam/dm-labs-test",
                eval_summary=eval_summary,
                comparison_summary=comparison_summary,
                eval_plan={"n_batches": 3, "T": 64, "timestep_grid": [1, 16, 32, 48, 64], "seed": 7, "batches": []},
                overwrite_model_card=True,
            )
            self.assertTrue(Path(bundle["manifest_path"]).exists())
            self.assertTrue(Path(bundle["eval_plan_path"]).exists())
            self.assertTrue(bundle["validation"]["ready_for_upload"])
            self.assertTrue(bundle["validation"]["checks"]["manifest_repo_id_matches"])
            validation = validate_hf_export_bundle(tmpdir, repo_id="bitlabsdevteam/dm-labs-test")
            self.assertTrue(validation["ready_for_upload"])
            self.assertTrue(validation["checks"]["eval_summary_exists"])
            self.assertTrue(validation["checks"]["schedule_comparison_exists"])
            card_path = ensure_hf_model_card(
                tmpdir,
                repo_id="bitlabsdevteam/dm-labs-test",
                eval_summary=eval_summary,
                comparison_summary=comparison_summary,
                overwrite=True,
            )
            content = Path(card_path).read_text(encoding="utf-8")

        self.assertIn("| timestep_uniform_sampled | uniform mean over sampled per-example timesteps | 1.1 | 2.1 | 16.0 | 3.1 | 0.8999999999999999 |", content)
        self.assertIn("| schedule_reweighted_sampled | inverse-expected-mask-ratio weighting over sampled masked tokens | 1.2 | 2.2 | 16.0 | 3.2 | 0.7999999999999998 |", content)
        self.assertIn("| token_weighted_sampled | token-weighted over sampled masked tokens | 1.0 | 2.0 | 16.0 | 3.0 | 1.0 |", content)
        self.assertIn("- uniform_random_pseudo_perplexity: 16.0", content)
        self.assertIn("- schedule_reweighted_nonzero_examples: 12", content)
        self.assertIn("- schedule_reweighted_effective_sample_size: 10.5", content)
        self.assertIn("- schedule_reweighted_aggregation: inverse_expected_mask_ratio_weighting_over_sampled_masked_tokens", content)
        self.assertIn("- optional `eval_plan.pt` shared cached evaluation plan artifact", content)
        self.assertIn("- optional `hf_export_manifest.json` bundle manifest covering all exported metadata files", content)
        self.assertIn("| schedule_reweighted_pseudo_perplexity | lower | 2.2 | 2.32 | 0.12 | cosine_schedule | 0.78 | False | True | -0.08 | 0.32 | 0.22 |", content)
        self.assertIn("| timestep_uniform_accuracy | higher | 0.41 | 0.389 | -0.021 | cosine_schedule | 0.89 | False | True | -0.041 | 0.001 | 0.11 |", content)
        self.assertIn("| timestep_auc_bits_saved_vs_uniform | higher | 0.5 | 0.482 | -0.018 | cosine_schedule | 0.75 | False | True | -0.055 | 0.025 | 0.75 |", content)
        self.assertIn("| sampled_denoising_skill | higher | 0.6393262397777592 | 0.6357195021755369 | -0.0036067376022224087 | cosine_schedule | 0.8 | False | True | -0.018 | 0.007 | 0.8 |", content)
        self.assertIn("| schedule_reweighted_accuracy | higher | 0.42 | 0.398 | -0.022 | cosine_schedule | 0.88 | False | True | -0.042 | 0.002 | 0.12 |", content)


if __name__ == "__main__":
    unittest.main()
