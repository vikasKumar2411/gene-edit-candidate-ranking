"""Tune the decision threshold using the validation split only."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.training.train_baseline import (
    assert_disjoint_groups,
    load_gold_dataset,
    select_model_features,
    split_by_group,
)


def main() -> None:
    """Select the best validation threshold and evaluate it on test data."""

    config = load_config()
    training = config["training"]
    threshold_config = training["threshold_tuning"]
    split_config = training["split"]

    dataframe = load_gold_dataset(config)

    train_df, validation_df, test_df = split_by_group(
        dataframe=dataframe,
        group_column=training["group_column"],
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        test_fraction=split_config["test_fraction"],
        random_seed=training["random_seed"],
    )

    assert_disjoint_groups(
        train=train_df,
        validation=validation_df,
        test=test_df,
        group_column=training["group_column"],
    )

    x_validation, y_validation = select_model_features(
        validation_df,
        training["target_column"],
        training["group_column"],
    )

    x_test, y_test = select_model_features(
        test_df,
        training["target_column"],
        training["group_column"],
    )

    model_path = (
        PROJECT_ROOT
        / training["output_dir"]
        / "model.joblib"
    )

    model = joblib.load(model_path)

    validation_probabilities = model.predict_proba(x_validation)[:, 1]

    thresholds = np.arange(
        threshold_config["minimum"],
        threshold_config["maximum"] + threshold_config["step"] / 2,
        threshold_config["step"],
    )

    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": float(round(threshold, 4)),
                "accuracy": float(
                    accuracy_score(y_validation, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_validation,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_validation,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_validation,
                        predictions,
                        zero_division=0,
                    )
                ),
            }
        )

    results = pd.DataFrame(rows)

    optimization_metric = threshold_config["optimization_metric"]

    best_row = (
        results
        .sort_values(
            by=[optimization_metric, "precision"],
            ascending=[False, False],
        )
        .iloc[0]
    )

    best_threshold = float(best_row["threshold"])

    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_metrics = {
        "threshold": best_threshold,
        "accuracy": float(
            accuracy_score(y_test, test_predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
    }

    output_dir = PROJECT_ROOT / training["output_dir"]

    results.to_csv(
        output_dir / "threshold_sweep.csv",
        index=False,
    )

    payload = {
        "optimization_metric": optimization_metric,
        "selected_threshold": best_threshold,
        "validation_metrics": {
            key: float(value)
            for key, value in best_row.to_dict().items()
        },
        "test_metrics": test_metrics,
    }

    with (
        output_dir / "threshold_selection.json"
    ).open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    print("Threshold tuning: SUCCESS")
    print(f"optimization_metric={optimization_metric}")
    print(f"selected_threshold={best_threshold:.2f}")
    print()
    print("Validation metrics:")
    for name, value in payload["validation_metrics"].items():
        print(f"  {name}: {value}")
    print()
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value}")
    print()
    print(
        "Sweep file:",
        output_dir / "threshold_sweep.csv",
    )
    print(
        "Selection file:",
        output_dir / "threshold_selection.json",
    )


if __name__ == "__main__":
    main()
