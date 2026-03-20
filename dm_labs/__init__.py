"""DM_Labs utilities for diffusion language model experiments."""

from .hf_utils import (
    build_comparison_protocol_rows,
    build_eval_protocol_rows,
    ensure_hf_model_card,
    render_eval_summary_report,
    render_hf_preflight_report,
    upload_checkpoint_to_hub,
    validate_hf_export_bundle,
    write_eval_summary,
    write_schedule_comparison,
)

__all__ = [
    "build_comparison_protocol_rows",
    "build_eval_protocol_rows",
    "ensure_hf_model_card",
    "render_eval_summary_report",
    "render_hf_preflight_report",
    "write_eval_summary",
    "write_schedule_comparison",
    "upload_checkpoint_to_hub",
    "validate_hf_export_bundle",
]

try:
    from .data_utils import TokenBlockDataset, collate_blocks, format_as_chat
    from .eval_utils import (
        build_eval_plan,
        compare_schedule_checkpoints,
        corrupt_with_mask,
        corruption_factory,
        evaluate_diffusion_pseudo_perplexity,
        evaluate_diffusion_pseudo_perplexity_from_plan,
        export_eval_result,
        export_schedule_comparison,
        load_diffusion_checkpoint,
        mask_ratio_cosine_schedule,
        mask_ratio_linear_schedule,
    )
    from .modeling import DiffusionLMConfig, DiffusionTransformerLM
except ImportError:
    # Allow lightweight helpers such as HF artifact/model-card utilities to be imported
    # in environments where torch is not installed.
    pass
else:
    __all__.extend(
        [
            "TokenBlockDataset",
            "collate_blocks",
            "format_as_chat",
            "DiffusionLMConfig",
            "DiffusionTransformerLM",
            "mask_ratio_cosine_schedule",
            "mask_ratio_linear_schedule",
            "corrupt_with_mask",
            "corruption_factory",
            "build_eval_plan",
            "evaluate_diffusion_pseudo_perplexity",
            "evaluate_diffusion_pseudo_perplexity_from_plan",
            "compare_schedule_checkpoints",
            "export_eval_result",
            "export_schedule_comparison",
            "load_diffusion_checkpoint",
        ]
    )
