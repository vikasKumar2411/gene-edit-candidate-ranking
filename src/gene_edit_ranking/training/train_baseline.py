"""Train and evaluate a leakage-safe logistic regression baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.gold_schema import GOLD_TRAINING_SCHEMA


@dataclass(frozen=True)
class SplitSummary:
    """Summary of one data split."""

    rows: int
    unique_candidates: int
    positive_rate: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Binary classification evaluation metrics."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def load_gold_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the local Gold training dataset."""

    gold_config = config["gold_dataset"]

    path = (
        PROJECT_ROOT
        / "data"
        / "gold"
        / "training"
        / f"dataset_version={gold_config['dataset_version']}"
        / f"processing_date={gold_config['processing_date']}"
        / "training_dataset.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {path}")

    return pd.read_parquet(path)


def split_by_group(
    dataframe: pd.DataFrame,
    group_column: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test splits with disjoint groups."""

    split_total = train_fraction + validation_fraction + test_fraction

    if not np.isclose(split_total, 1.0):
        raise ValueError(
            f"Split fractions must total 1.0; found {split_total}."
        )

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=random_seed,
    )

    train_indices, temporary_indices = next(
        first_split.split(
            dataframe,
            groups=dataframe[group_column],
        )
    )

    train = dataframe.iloc[train_indices].reset_index(drop=True)
    temporary = dataframe.iloc[temporary_indices].reset_index(drop=True)

    validation_share_of_temporary = (
        validation_fraction / (validation_fraction + test_fraction)
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_share_of_temporary,
        random_state=random_seed + 1,
    )

    validation_indices, test_indices = next(
        second_split.split(
            temporary,
            groups=temporary[group_column],
        )
    )

    validation = temporary.iloc[validation_indices].reset_index(drop=True)
    test = temporary.iloc[test_indices].reset_index(drop=True)

    return train, validation, test


def assert_disjoint_groups(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    group_column: str,
) -> None:
    """Verify that no candidate group crosses data splits."""

    train_groups = set(train[group_column])
    validation_groups = set(validation[group_column])
    test_groups = set(test[group_column])

    overlaps = {
        "train_validation": train_groups & validation_groups,
        "train_test": train_groups & test_groups,
        "validation_test": validation_groups & test_groups,
    }

    failing_overlaps = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if failing_overlaps:
        raise RuntimeError(
            f"Group leakage detected across splits: {failing_overlaps}"
        )


def select_model_features(
    dataframe: pd.DataFrame,
    target_column: str,
    group_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select model inputs while excluding identifiers and leakage columns."""

    excluded_columns = {
        target_column,
        group_column,
        *GOLD_TRAINING_SCHEMA.entity_columns,
        *GOLD_TRAINING_SCHEMA.target_columns,
        "created_date",
        "experiment_date",
    }

    excluded_columns.update(
        column
        for column in dataframe.columns
        if column.startswith("_")
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No model feature columns remain after exclusions.")

    return dataframe[feature_columns].copy(), dataframe[target_column].copy()


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    training_config: dict[str, Any],
) -> Pipeline:
    """Build preprocessing and logistic regression as one fitted pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    logistic_config = training_config["logistic_regression"]

    classifier = LogisticRegression(
        max_iter=logistic_config["max_iter"],
        class_weight=logistic_config["class_weight"],
        solver=logistic_config["solver"],
        random_state=training_config["random_seed"],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> EvaluationMetrics:
    """Evaluate predicted probabilities at a configured threshold."""

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    return EvaluationMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        roc_auc=float(roc_auc_score(target, probabilities)),
        pr_auc=float(average_precision_score(target, probabilities)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def summarize_split(
    dataframe: pd.DataFrame,
    target_column: str,
    group_column: str,
) -> SplitSummary:
    """Create a concise data split summary."""

    return SplitSummary(
        rows=len(dataframe),
        unique_candidates=int(dataframe[group_column].nunique()),
        positive_rate=float(dataframe[target_column].mean()),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train, evaluate, and persist the local baseline."""

    training_config = config["training"]
    target_column = training_config["target_column"]
    group_column = training_config["group_column"]
    split_config = training_config["split"]

    dataframe = load_gold_dataset(config)

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

    pipeline = build_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        training_config=training_config,
    )

    pipeline.fit(x_train, y_train)

    threshold = training_config["decision_threshold"]

    validation_metrics = evaluate(
        model=pipeline,
        features=x_validation,
        target=y_validation,
        threshold=threshold,
    )

    test_metrics = evaluate(
        model=pipeline,
        features=x_test,
        target=y_test,
        threshold=threshold,
    )

    output_dir = PROJECT_ROOT / training_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "evaluation.json"
    feature_manifest_path = output_dir / "feature_manifest.json"

    joblib.dump(pipeline, model_path)

    evaluation_payload = {
        "model_name": training_config["model_name"],
        "model_version": training_config["model_version"],
        "target_column": target_column,
        "group_column": group_column,
        "splits": {
            "train": asdict(
                summarize_split(
                    train_df,
                    target_column,
                    group_column,
                )
            ),
            "validation": asdict(
                summarize_split(
                    validation_df,
                    target_column,
                    group_column,
                )
            ),
            "test": asdict(
                summarize_split(
                    test_df,
                    target_column,
                    group_column,
                )
            ),
        },
        "validation_metrics": asdict(validation_metrics),
        "test_metrics": asdict(test_metrics),
    }

    write_json(metrics_path, evaluation_payload)

    write_json(
        feature_manifest_path,
        {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "excluded_identifier_columns": list(
                GOLD_TRAINING_SCHEMA.entity_columns
            ),
            "excluded_target_columns": list(
                GOLD_TRAINING_SCHEMA.target_columns
            ),
        },
    )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_manifest_path": str(feature_manifest_path),
        **evaluation_payload,
    }


def main() -> None:
    """Run local baseline training."""

    config = load_config()
    result = train(config)

    print("Local baseline training: SUCCESS")
    print()

    for split_name, split_summary in result["splits"].items():
        print(
            f"{split_name:<10} "
            f"rows={split_summary['rows']:>4} "
            f"candidates={split_summary['unique_candidates']:>4} "
            f"positive_rate={split_summary['positive_rate']:.4f}"
        )

    print()
    print("Validation metrics:")
    for name, value in result["validation_metrics"].items():
        print(f"  {name}: {value}")

    print()
    print("Test metrics:")
    for name, value in result["test_metrics"].items():
        print(f"  {name}: {value}")

    print()
    print(f"Model artifact: {result['model_path']}")
    print(f"Evaluation:     {result['metrics_path']}")
    print(f"Features:       {result['feature_manifest_path']}")


if __name__ == "__main__":
    main()
