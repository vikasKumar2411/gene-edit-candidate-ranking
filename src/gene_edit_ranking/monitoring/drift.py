"""Lightweight feature-drift monitoring utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def normalized_mean_shift(
    *,
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Return mean shift measured in reference standard deviations."""

    reference_numeric = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if reference_numeric.empty or current_numeric.empty:
        return 0.0

    reference_std = float(reference_numeric.std(ddof=0))

    mean_difference = abs(
        float(current_numeric.mean())
        - float(reference_numeric.mean())
    )

    if reference_std == 0:
        return 0.0 if mean_difference == 0 else float("inf")

    return mean_difference / reference_std


def categorical_distribution_shift(
    *,
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Return the largest absolute categorical frequency change."""

    reference_distribution = (
        reference.fillna("__MISSING__")
        .astype(str)
        .value_counts(normalize=True)
    )

    current_distribution = (
        current.fillna("__MISSING__")
        .astype(str)
        .value_counts(normalize=True)
    )

    categories = (
        set(reference_distribution.index)
        | set(current_distribution.index)
    )

    if not categories:
        return 0.0

    return max(
        abs(
            float(reference_distribution.get(category, 0.0))
            - float(current_distribution.get(category, 0.0))
        )
        for category in categories
    )


def analyze_feature_drift(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_shift_threshold: float = 0.5,
    categorical_shift_threshold: float = 0.2,
) -> dict[str, Any]:
    """Compare current model features against a reference dataset."""

    model_columns = [
        *numeric_columns,
        *categorical_columns,
    ]

    missing_columns = [
        column
        for column in model_columns
        if column not in current.columns
    ]

    extra_columns = sorted(
        set(current.columns) - set(reference.columns)
    )

    numeric_results: dict[str, Any] = {}

    for column in numeric_columns:
        if column not in reference.columns or column not in current.columns:
            continue

        shift = normalized_mean_shift(
            reference=reference[column],
            current=current[column],
        )

        numeric_results[column] = {
            "reference_mean": float(
                pd.to_numeric(
                    reference[column],
                    errors="coerce",
                ).mean()
            ),
            "current_mean": float(
                pd.to_numeric(
                    current[column],
                    errors="coerce",
                ).mean()
            ),
            "normalized_mean_shift": shift,
            "drift_detected": shift >= numeric_shift_threshold,
        }

    categorical_results: dict[str, Any] = {}

    for column in categorical_columns:
        if column not in reference.columns or column not in current.columns:
            continue

        shift = categorical_distribution_shift(
            reference=reference[column],
            current=current[column],
        )

        categorical_results[column] = {
            "max_frequency_shift": shift,
            "drift_detected": shift >= categorical_shift_threshold,
        }

    drifted_numeric = [
        column
        for column, result in numeric_results.items()
        if result["drift_detected"]
    ]

    drifted_categorical = [
        column
        for column, result in categorical_results.items()
        if result["drift_detected"]
    ]

    overall_drift_detected = bool(
        missing_columns
        or drifted_numeric
        or drifted_categorical
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "row_count_ratio": (
            len(current) / len(reference)
            if len(reference) > 0
            else None
        ),
        "thresholds": {
            "numeric_normalized_mean_shift": numeric_shift_threshold,
            "categorical_frequency_shift": categorical_shift_threshold,
        },
        "schema": {
            "missing_model_columns": missing_columns,
            "extra_columns": extra_columns,
        },
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
        "summary": {
            "numeric_features_evaluated": len(numeric_results),
            "categorical_features_evaluated": len(
                categorical_results
            ),
            "drifted_numeric_features": drifted_numeric,
            "drifted_categorical_features": drifted_categorical,
            "overall_drift_detected": overall_drift_detected,
        },
    }


def write_drift_report(
    *,
    report: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a drift report as formatted JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return output_path
