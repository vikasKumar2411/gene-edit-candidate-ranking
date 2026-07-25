"""Contracts for reusable offline feature datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    """Defines the contract for one reusable feature table."""

    entity_key: str
    required_columns: tuple[str, ...]
    embedding_prefix: str
    embedding_dimension: int


def build_feature_schemas(
    gene_embedding_dimension: int,
    crop_line_embedding_dimension: int,
    environment_embedding_dimension: int,
) -> dict[str, FeatureSchema]:
    """Build reusable feature contracts from configured dimensions."""

    return {
        "gene_features": FeatureSchema(
            entity_key="gene_id",
            required_columns=(
                "gene_id",
                "synthetic_conservation_score",
                "gene_length",
                "normalized_gene_position",
            ),
            embedding_prefix="gene_embedding",
            embedding_dimension=gene_embedding_dimension,
        ),
        "crop_line_features": FeatureSchema(
            entity_key="crop_line_id",
            required_columns=(
                "crop_line_id",
                "baseline_yield_index",
                "baseline_drought_tolerance",
                "yield_drought_interaction",
            ),
            embedding_prefix="crop_line_embedding",
            embedding_dimension=crop_line_embedding_dimension,
        ),
        "environment_features": FeatureSchema(
            entity_key="environment_id",
            required_columns=(
                "environment_id",
                "rainfall_mm",
                "average_temperature_c",
                "synthetic_drought_index",
                "environment_stress_score",
            ),
            embedding_prefix="environment_embedding",
            embedding_dimension=environment_embedding_dimension,
        ),
    }
