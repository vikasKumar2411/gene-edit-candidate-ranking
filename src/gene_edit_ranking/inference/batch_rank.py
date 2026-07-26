"""Batch-score and rank gene-edit candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class BatchRankingSummary:
    """Summary metadata for a batch-ranking run."""

    input_rows: int
    output_rows: int
    model_package_version: int
    model_package_arn: str
    model_data_uri: str
    input_data_uri: str
    selected_threshold: float
    ranking_columns: list[str]
    aggregation_method: str
    generated_at_utc: str


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the model feature manifest."""

    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Ensure all required model features exist."""

    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Input data is missing model features: {missing}"
        )


def rank_candidates(
    *,
    dataframe: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    threshold: float,
    ranking_columns: list[str],
) -> pd.DataFrame:
    """Score and rank candidates within business partitions."""

    validate_model_features(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    scored = dataframe.copy()

    probabilities = model.predict_proba(
        scored[feature_columns]
    )[:, 1]

    scored["edit_success_probability"] = probabilities
    scored["predicted_edit_success"] = (
        probabilities >= threshold
    ).astype(int)

    candidate_keys = [
        "candidate_id",
        *ranking_columns,
    ]

    candidate_scores = (
        scored.groupby(
            candidate_keys,
            dropna=False,
            as_index=False,
        )
        .agg(
            edit_success_probability=(
                "edit_success_probability",
                "mean",
            ),
            experiment_count=(
                "edit_success_probability",
                "size",
            ),
        )
    )

    candidate_scores["predicted_edit_success"] = (
        candidate_scores["edit_success_probability"]
        >= threshold
    ).astype(int)

    candidate_scores["rank_within_group"] = (
        candidate_scores.groupby(
            ranking_columns,
            dropna=False,
        )["edit_success_probability"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    candidate_scores = candidate_scores.sort_values(
        by=[
            *ranking_columns,
            "rank_within_group",
        ],
        ascending=True,
    ).reset_index(drop=True)

    return candidate_scores


def write_batch_outputs(
    *,
    scored: pd.DataFrame,
    output_directory: Path,
    summary: BatchRankingSummary,
) -> tuple[Path, Path]:
    """Write ranked predictions and run metadata."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_directory
        / "ranked_candidates.parquet"
    )

    metadata_path = (
        output_directory
        / "run_metadata.json"
    )

    scored.to_parquet(
        predictions_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    return predictions_path, metadata_path


def run_batch_ranking(
    *,
    input_path: Path,
    model_directory: Path,
    output_directory: Path,
    model_package_version: int,
    model_package_arn: str,
    model_data_uri: str,
    input_data_uri: str,
    ranking_columns: list[str],
) -> tuple[pd.DataFrame, BatchRankingSummary]:
    """Run batch scoring and ranking."""

    model_path = model_directory / "model.joblib"
    manifest_path = (
        model_directory
        / "feature_manifest.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing feature manifest: {manifest_path}"
        )

    dataframe = pd.read_parquet(input_path)
    model = joblib.load(model_path)
    manifest = load_manifest(manifest_path)

    feature_columns = list(model.feature_names_in_)
    threshold = float(
        manifest["selected_threshold"]
    )

    scored = rank_candidates(
        dataframe=dataframe,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        ranking_columns=ranking_columns,
    )

    summary = BatchRankingSummary(
        input_rows=len(dataframe),
        output_rows=len(scored),
        model_package_version=model_package_version,
        model_package_arn=model_package_arn,
        model_data_uri=model_data_uri,
        input_data_uri=input_data_uri,
        selected_threshold=threshold,
        ranking_columns=ranking_columns,
        aggregation_method=(
            "mean probability across experiments per "
            "candidate and ranking context"
        ),
        generated_at_utc=datetime.now(
            UTC
        ).isoformat(),
    )

    write_batch_outputs(
        scored=scored,
        output_directory=output_directory,
        summary=summary,
    )

    return scored, summary
