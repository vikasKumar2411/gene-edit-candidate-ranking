"""Package the trained baseline and its metadata into a deployable archive."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from gene_edit_ranking.config import PROJECT_ROOT, load_config


REQUIRED_FILES = (
    "model.joblib",
    "evaluation.json",
    "feature_manifest.json",
    "threshold_selection.json",
)


def calculate_sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document."""

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def main() -> None:
    """Validate and package the trained model and supporting metadata."""

    config = load_config()
    training = config["training"]
    gold = config["gold_dataset"]
    features = config["feature_generation"]

    training_dir = PROJECT_ROOT / training["output_dir"]

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (training_dir / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required training artifacts: {missing_files}"
        )

    model_path = training_dir / "model.joblib"
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support probability predictions."
        )

    evaluation = load_json(training_dir / "evaluation.json")
    threshold_selection = load_json(
        training_dir / "threshold_selection.json"
    )
    feature_manifest = load_json(
        training_dir / "feature_manifest.json"
    )

    model_metadata = {
        "model_name": training["model_name"],
        "model_version": training["model_version"],
        "model_type": "sklearn.pipeline.Pipeline",
        "task_type": "binary_classification",
        "target_column": training["target_column"],
        "group_column": training["group_column"],
        "selected_threshold": threshold_selection[
            "selected_threshold"
        ],
        "threshold_optimization_metric": threshold_selection[
            "optimization_metric"
        ],
        "gold_dataset_version": gold["dataset_version"],
        "feature_version": features["feature_version"],
        "source_version": config["data"]["source_version"],
        "python_version": (
            f"{__import__('sys').version_info.major}."
            f"{__import__('sys').version_info.minor}."
            f"{__import__('sys').version_info.micro}"
        ),
        "scikit_learn_version": sklearn.__version__,
        "numeric_feature_count": len(
            feature_manifest["numeric_columns"]
        ),
        "categorical_feature_count": len(
            feature_manifest["categorical_columns"]
        ),
        "validation_metrics": evaluation["validation_metrics"],
        "test_metrics_at_default_threshold": evaluation["test_metrics"],
        "test_metrics_at_selected_threshold": threshold_selection[
            "test_metrics"
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic": True,
        "disclaimer": (
            "Synthetic portfolio model only. "
            "It makes no real biological claims."
        ),
    }

    metadata_path = training_dir / "model_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(model_metadata, file_handle, indent=2)

    bundle_dir = (
        training_dir
        / training["model_name"]
        / f"model_version={training['model_version']}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = bundle_dir / "model.tar.gz"

    bundle_files = (
        *REQUIRED_FILES,
        "model_metadata.json",
    )

    with tarfile.open(bundle_path, mode="w:gz") as archive:
        for filename in bundle_files:
            archive.add(
                training_dir / filename,
                arcname=filename,
            )

    checksum = calculate_sha256(bundle_path)

    checksum_path = bundle_dir / "model.tar.gz.sha256"
    checksum_path.write_text(
        f"{checksum}  model.tar.gz\n",
        encoding="utf-8",
    )

    print("Model packaging: SUCCESS")
    print(f"bundle={bundle_path}")
    print(f"size_bytes={bundle_path.stat().st_size}")
    print(f"sha256={checksum}")
    print(
        "selected_threshold="
        f"{model_metadata['selected_threshold']}"
    )
    print(
        "test_f1_at_selected_threshold="
        f"{model_metadata['test_metrics_at_selected_threshold']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
