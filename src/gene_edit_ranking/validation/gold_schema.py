"""Contract for the model-ready Gold training dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldDatasetSchema:
    """Defines the expected Gold training dataset structure."""

    primary_key: str
    entity_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    required_numeric_columns: tuple[str, ...]
    required_categorical_columns: tuple[str, ...]


GOLD_TRAINING_SCHEMA = GoldDatasetSchema(
    primary_key="experiment_id",
    entity_columns=(
        "experiment_id",
        "candidate_id",
        "gene_id",
        "crop_line_id",
        "environment_id",
    ),
    target_columns=(
        "observed_edit_success",
        "yield_change_percent",
        "drought_response_score",
    ),
    required_numeric_columns=(
        "replicate_number",
        "predicted_edit_efficiency",
        "predicted_off_target_risk",
        "target_position",
        "measurement_quality_score",
        "synthetic_conservation_score",
        "gene_length",
        "normalized_gene_position",
        "baseline_yield_index",
        "baseline_drought_tolerance",
        "yield_drought_interaction",
        "rainfall_mm",
        "average_temperature_c",
        "synthetic_drought_index",
        "environment_stress_score",
    ),
    required_categorical_columns=(
        "editing_method",
        "design_batch",
    ),
)
