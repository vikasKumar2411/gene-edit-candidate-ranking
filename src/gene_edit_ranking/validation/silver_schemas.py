"""Column contracts for normalized Silver datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSchema:
    """Defines the expected Silver contract for one dataset."""

    primary_key: str
    string_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    float_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    date_columns: tuple[str, ...]


SILVER_SCHEMAS: dict[str, DatasetSchema] = {
    "genes": DatasetSchema(
        primary_key="gene_id",
        string_columns=(
            "gene_id",
            "gene_symbol",
            "chromosome",
            "gene_family",
        ),
        integer_columns=(
            "start_position",
            "end_position",
        ),
        float_columns=(
            "synthetic_conservation_score",
        ),
        boolean_columns=(
            "is_synthetic",
        ),
        date_columns=(),
    ),
    "crop_lines": DatasetSchema(
        primary_key="crop_line_id",
        string_columns=(
            "crop_line_id",
            "crop_type",
            "breeding_program",
            "maturity_group",
        ),
        integer_columns=(),
        float_columns=(
            "baseline_yield_index",
            "baseline_drought_tolerance",
        ),
        boolean_columns=(
            "is_synthetic",
        ),
        date_columns=(),
    ),
    "environments": DatasetSchema(
        primary_key="environment_id",
        string_columns=(
            "environment_id",
            "location_code",
            "season",
            "soil_type",
        ),
        integer_columns=(),
        float_columns=(
            "rainfall_mm",
            "average_temperature_c",
            "synthetic_drought_index",
        ),
        boolean_columns=(
            "is_synthetic",
        ),
        date_columns=(),
    ),
    "candidates": DatasetSchema(
        primary_key="candidate_id",
        string_columns=(
            "candidate_id",
            "gene_id",
            "crop_line_id",
            "editing_method",
            "design_batch",
        ),
        integer_columns=(
            "target_position",
        ),
        float_columns=(
            "predicted_edit_efficiency",
            "predicted_off_target_risk",
        ),
        boolean_columns=(
            "is_synthetic",
        ),
        date_columns=(
            "created_date",
        ),
    ),
    "experiments": DatasetSchema(
        primary_key="experiment_id",
        string_columns=(
            "experiment_id",
            "candidate_id",
            "environment_id",
        ),
        integer_columns=(
            "replicate_number",
            "observed_edit_success",
        ),
        float_columns=(
            "yield_change_percent",
            "drought_response_score",
            "measurement_quality_score",
        ),
        boolean_columns=(
            "is_synthetic",
        ),
        date_columns=(
            "experiment_date",
        ),
    ),
}
