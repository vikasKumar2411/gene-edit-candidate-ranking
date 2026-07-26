"""Compare a scoring dataset against the training reference dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gene_edit_ranking.monitoring.drift import (
    analyze_feature_drift,
    write_drift_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--current-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--feature-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/monitoring/drift_report.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference = pd.read_parquet(args.reference_data)
    current = pd.read_parquet(args.current_data)

    manifest = json.loads(
        args.feature_manifest.read_text(encoding="utf-8")
    )

    report = analyze_feature_drift(
        reference=reference,
        current=current,
        numeric_columns=manifest["numeric_columns"],
        categorical_columns=manifest["categorical_columns"],
    )

    write_drift_report(
        report=report,
        output_path=args.output,
    )

    summary = report["summary"]

    print(f"drift_report={args.output}")
    print(
        "overall_drift_detected="
        f"{summary['overall_drift_detected']}"
    )
    print(
        "drifted_numeric_features="
        f"{len(summary['drifted_numeric_features'])}"
    )
    print(
        "drifted_categorical_features="
        f"{len(summary['drifted_categorical_features'])}"
    )


if __name__ == "__main__":
    main()
