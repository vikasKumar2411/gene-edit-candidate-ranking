"""SageMaker training entry point for the edit-success baseline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gene_edit_ranking.config import load_config
from gene_edit_ranking.training.train_baseline import (
    assert_disjoint_groups,
    build_pipeline,
    evaluate,
    select_model_features,
    split_by_group,
    summarize_split,
)
from gene_edit_ranking.validation.gold_schema import GOLD_TRAINING_SCHEMA

TRAINING_CHANNEL = Path(
    os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
)
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path(
    os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
)


def find_training_dataset() -> Path:
    """Find the Gold Parquet file staged by SageMaker."""

    parquet_files = sorted(TRAINING_CHANNEL.rglob("*.parquet"))

    if len(parquet_files) != 1:
        raise RuntimeError(
            "Expected exactly one training Parquet file under "
            f"{TRAINING_CHANNEL}; found {len(parquet_files)}."
        )

    return parquet_files[0]


def select_threshold(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[float, list[dict[str, float]]]:
    """Choose the validation threshold with maximum F1."""

    probabilities = model.predict_proba(features)[:, 1]
    threshold_rows: list[dict[str, float]] = []

    thresholds = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        threshold_rows.append(
            {
                "threshold": float(round(threshold, 4)),
                "accuracy": float(
                    accuracy_score(target, predictions)
                ),
                "precision": float(
                    precision_score(
                        target,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        target,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        target,
                        predictions,
                        zero_division=0,
                    )
                ),
            }
        )

    best_row = sorted(
        threshold_rows,
        key=lambda row: (row["f1"], row["precision"]),
        reverse=True,
    )[0]

    return best_row["threshold"], threshold_rows


def main() -> None:
    """Train, evaluate, and write SageMaker model artifacts."""

    config = load_config()
    training_config = config["training"]
    split_config = training_config["split"]
    threshold_config = training_config["threshold_tuning"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_training_dataset()
    dataframe = pd.read_parquet(dataset_path)

    target_column = training_config["target_column"]
    group_column = training_config["group_column"]

    train_df, validation_df, test_df = split_by_group(
        dataframe=dataframe,
        group_column=group_column,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
        random_seed=training_config["random_seed"],
    )

    assert_disjoint_groups(
        train=train_df,
        validation=validation_df,
        test=test_df,
        group_column=group_column,
    )

    x_train, y_train = select_model_features(
        train_df,
        target_column,
        group_column,
    )
    x_validation, y_validation = select_model_features(
        validation_df,
        target_column,
        group_column,
    )
    x_test, y_test = select_model_features(
        test_df,
        target_column,
        group_column,
    )

    categorical_columns = [
        column
        for column in GOLD_TRAINING_SCHEMA.required_categorical_columns
        if column in x_train.columns
    ]

    numeric_columns = [
        column
        for column in x_train.columns
        if column not in categorical_columns
        and pd.api.types.is_numeric_dtype(x_train[column])
    ]

    model = build_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        training_config=training_config,
    )

    model.fit(x_train, y_train)

    selected_threshold, threshold_rows = select_threshold(
        model=model,
        features=x_validation,
        target=y_validation,
        minimum=threshold_config["minimum"],
        maximum=threshold_config["maximum"],
        step=threshold_config["step"],
    )

    validation_metrics = evaluate(
        model=model,
        features=x_validation,
        target=y_validation,
        threshold=selected_threshold,
    )

    test_metrics = evaluate(
        model=model,
        features=x_test,
        target=y_test,
        threshold=selected_threshold,
    )

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    evaluation = {
        "model_name": training_config["model_name"],
        "model_version": training_config["model_version"],
        "target_column": target_column,
        "group_column": group_column,
        "selected_threshold": selected_threshold,
        "splits": {
            "train": asdict(
                summarize_split(train_df, target_column, group_column)
            ),
            "validation": asdict(
                summarize_split(validation_df, target_column, group_column)
            ),
            "test": asdict(
                summarize_split(test_df, target_column, group_column)
            ),
        },
        "validation_metrics": asdict(validation_metrics),
        "test_metrics": asdict(test_metrics),
        "validation_roc_auc": float(
            roc_auc_score(y_validation, validation_probabilities)
        ),
        "validation_pr_auc": float(
            average_precision_score(
                y_validation,
                validation_probabilities,
            )
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
        "test_pr_auc": float(
            average_precision_score(y_test, test_probabilities)
        ),
    }

    feature_manifest = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "selected_threshold": selected_threshold,
    }

    joblib.dump(model, MODEL_DIR / "model.joblib")

    (MODEL_DIR / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2),
        encoding="utf-8",
    )

    (MODEL_DIR / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / "threshold_sweep.csv",
        index=False,
    )

    print("SageMaker training: SUCCESS")
    print(f"training_dataset={dataset_path}")
    print(f"selected_threshold={selected_threshold:.2f}")
    print(f"validation_pr_auc={evaluation['validation_pr_auc']:.6f}")
    print(f"test_pr_auc={evaluation['test_pr_auc']:.6f}")
    print(f"test_roc_auc={evaluation['test_roc_auc']:.6f}")
    print(f"test_f1={test_metrics.f1:.6f}")


if __name__ == "__main__":
    main()
