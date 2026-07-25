"""Independent SageMaker Processing evaluation entry point."""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from gene_edit_ranking.config import load_config
from gene_edit_ranking.training.train_baseline import (
    assert_disjoint_groups,
    evaluate,
    select_model_features,
    split_by_group,
    summarize_split,
)


MODEL_INPUT_DIR = Path(
    os.environ.get(
        "MODEL_INPUT_DIR",
        "/opt/ml/processing/model",
    )
)

DATA_INPUT_DIR = Path(
    os.environ.get(
        "DATA_INPUT_DIR",
        "/opt/ml/processing/data",
    )
)

EVALUATION_OUTPUT_DIR = Path(
    os.environ.get(
        "EVALUATION_OUTPUT_DIR",
        "/opt/ml/processing/evaluation",
    )
)


def find_exactly_one(
    directory: Path,
    pattern: str,
    description: str,
) -> Path:
    """Find exactly one required input file."""

    matches = sorted(directory.rglob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} under "
            f"{directory}; found {len(matches)}."
        )

    return matches[0]


def extract_model_bundle(
    model_archive: Path,
    destination: Path,
) -> None:
    """Extract the SageMaker model archive."""

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(model_archive, mode="r:gz") as archive:
        archive.extractall(destination, filter="data")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document."""

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def main() -> None:
    """Evaluate the trained model on the deterministic test split."""

    config = load_config()
    training_config = config["training"]
    split_config = training_config["split"]

    model_archive = find_exactly_one(
        MODEL_INPUT_DIR,
        "model.tar.gz",
        "model archive",
    )

    gold_dataset_path = find_exactly_one(
        DATA_INPUT_DIR,
        "*.parquet",
        "Gold Parquet dataset",
    )

    extracted_model_dir = Path("/tmp/extracted-model")
    extract_model_bundle(
        model_archive=model_archive,
        destination=extracted_model_dir,
    )

    model_path = extracted_model_dir / "model.joblib"
    manifest_path = extracted_model_dir / "feature_manifest.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact missing after extraction: {model_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Feature manifest missing after extraction: {manifest_path}"
        )

    model = joblib.load(model_path)
    manifest = load_json(manifest_path)

    selected_threshold = float(
        manifest["selected_threshold"]
    )

    dataframe = pd.read_parquet(gold_dataset_path)

    train_df, validation_df, test_df = split_by_group(
        dataframe=dataframe,
        group_column=training_config["group_column"],
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
        random_seed=training_config["random_seed"],
    )

    assert_disjoint_groups(
        train=train_df,
        validation=validation_df,
        test=test_df,
        group_column=training_config["group_column"],
    )

    x_test, y_test = select_model_features(
        dataframe=test_df,
        target_column=training_config["target_column"],
        group_column=training_config["group_column"],
    )

    metrics = evaluate(
        model=model,
        features=x_test,
        target=y_test,
        threshold=selected_threshold,
    )

    evaluation_report = {
        "model_name": training_config["model_name"],
        "model_version": training_config["model_version"],
        "target_column": training_config["target_column"],
        "selected_threshold": selected_threshold,
        "test_split": asdict(
            summarize_split(
                dataframe=test_df,
                target_column=training_config["target_column"],
                group_column=training_config["group_column"],
            )
        ),
        "test_metrics": asdict(metrics),
        "quality_gate_metric": {
            "name": "test_pr_auc",
            "value": metrics.pr_auc,
        },
        "is_synthetic": True,
        "disclaimer": (
            "Synthetic portfolio evaluation only. "
            "It makes no real biological claims."
        ),
    }

    EVALUATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        EVALUATION_OUTPUT_DIR
        / "evaluation.json"
    )

    output_path.write_text(
        json.dumps(evaluation_report, indent=2),
        encoding="utf-8",
    )

    print("Independent model evaluation: SUCCESS")
    print(f"model_archive={model_archive}")
    print(f"gold_dataset={gold_dataset_path}")
    print(f"selected_threshold={selected_threshold:.2f}")
    print(f"test_pr_auc={metrics.pr_auc:.6f}")
    print(f"test_roc_auc={metrics.roc_auc:.6f}")
    print(f"test_f1={metrics.f1:.6f}")
    print(f"evaluation_output={output_path}")


if __name__ == "__main__":
    main()
