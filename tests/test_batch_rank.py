"""Tests for batch candidate scoring and ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gene_edit_ranking.inference.batch_rank import (
    rank_candidates,
    validate_model_features,
)


class FakeProbabilityModel:
    """Return deterministic probabilities from the feature value."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = features["feature_score"].to_numpy(dtype=float)

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )


def test_rank_candidates_aggregates_experiments_to_unique_candidate_grain():
    dataframe = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c2", "c2"],
            "crop_line_id": ["line1", "line1", "line1", "line1"],
            "environment_id": ["env1", "env1", "env1", "env1"],
            "feature_score": [0.8, 0.6, 0.5, 0.3],
        }
    )

    result = rank_candidates(
        dataframe=dataframe,
        model=FakeProbabilityModel(),
        feature_columns=["feature_score"],
        threshold=0.5,
        ranking_columns=[
            "crop_line_id",
            "environment_id",
        ],
    )

    assert len(result) == 2

    assert not result.duplicated(
        subset=[
            "candidate_id",
            "crop_line_id",
            "environment_id",
        ]
    ).any()

    candidate_one = result.loc[
        result["candidate_id"] == "c1"
    ].iloc[0]

    candidate_two = result.loc[
        result["candidate_id"] == "c2"
    ].iloc[0]

    assert candidate_one["edit_success_probability"] == pytest.approx(0.7)
    assert candidate_one["experiment_count"] == 2
    assert candidate_one["predicted_edit_success"] == 1
    assert candidate_one["rank_within_group"] == 1

    assert candidate_two["edit_success_probability"] == pytest.approx(0.4)
    assert candidate_two["experiment_count"] == 2
    assert candidate_two["predicted_edit_success"] == 0
    assert candidate_two["rank_within_group"] == 2


def test_rank_candidates_ranks_independently_within_each_context():
    dataframe = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "crop_line_id": ["line1", "line1", "line2", "line2"],
            "environment_id": ["env1", "env1", "env1", "env1"],
            "feature_score": [0.9, 0.7, 0.6, 0.8],
        }
    )

    result = rank_candidates(
        dataframe=dataframe,
        model=FakeProbabilityModel(),
        feature_columns=["feature_score"],
        threshold=0.5,
        ranking_columns=[
            "crop_line_id",
            "environment_id",
        ],
    )

    line_one = result[result["crop_line_id"] == "line1"]
    line_two = result[result["crop_line_id"] == "line2"]

    assert line_one["rank_within_group"].tolist() == [1, 2]
    assert line_one["candidate_id"].tolist() == ["c1", "c2"]

    assert line_two["rank_within_group"].tolist() == [1, 2]
    assert line_two["candidate_id"].tolist() == ["c4", "c3"]


def test_validate_model_features_raises_for_missing_features():
    dataframe = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "feature_a": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Input data is missing model features",
    ):
        validate_model_features(
            dataframe=dataframe,
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
        )


def test_threshold_is_inclusive():
    dataframe = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "crop_line_id": ["line1"],
            "environment_id": ["env1"],
            "feature_score": [0.5],
        }
    )

    result = rank_candidates(
        dataframe=dataframe,
        model=FakeProbabilityModel(),
        feature_columns=["feature_score"],
        threshold=0.5,
        ranking_columns=[
            "crop_line_id",
            "environment_id",
        ],
    )

    assert result.iloc[0]["predicted_edit_success"] == 1


class PersistedFakeModel:
    """Pickle-friendly model used by run_batch_ranking tests."""

    feature_names_in_ = np.array(["feature_score"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = features["feature_score"].to_numpy(dtype=float)

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )


def test_run_batch_ranking_writes_predictions_and_metadata(tmp_path):
    import json

    import joblib

    from gene_edit_ranking.inference.batch_rank import run_batch_ranking

    input_path = tmp_path / "input.parquet"
    model_directory = tmp_path / "model"
    output_directory = tmp_path / "output"

    model_directory.mkdir()

    dataframe = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c2"],
            "crop_line_id": ["line1", "line1", "line1"],
            "environment_id": ["env1", "env1", "env1"],
            "feature_score": [0.8, 0.6, 0.4],
        }
    )

    dataframe.to_parquet(input_path, index=False)

    joblib.dump(
        PersistedFakeModel(),
        model_directory / "model.joblib",
    )

    (model_directory / "feature_manifest.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    scored, summary = run_batch_ranking(
        input_path=input_path,
        model_directory=model_directory,
        output_directory=output_directory,
        model_package_version=2,
        model_package_arn="arn:aws:sagemaker:test:model-package/2",
        model_data_uri="s3://test-bucket/model.tar.gz",
        input_data_uri="s3://test-bucket/input.parquet",
        ranking_columns=[
            "crop_line_id",
            "environment_id",
        ],
    )

    assert len(scored) == 2
    assert summary.input_rows == 3
    assert summary.output_rows == 2
    assert summary.model_package_version == 2
    assert summary.selected_threshold == 0.5

    predictions_path = (
        output_directory / "ranked_candidates.parquet"
    )

    metadata_path = (
        output_directory / "run_metadata.json"
    )

    assert predictions_path.exists()
    assert metadata_path.exists()

    written_predictions = pd.read_parquet(predictions_path)
    written_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert len(written_predictions) == 2
    assert written_metadata["input_rows"] == 3
    assert written_metadata["output_rows"] == 2
    assert written_metadata["model_package_version"] == 2


def test_run_batch_ranking_raises_when_model_file_is_missing(tmp_path):
    import json

    from gene_edit_ranking.inference.batch_rank import run_batch_ranking

    input_path = tmp_path / "input.parquet"
    model_directory = tmp_path / "model"
    output_directory = tmp_path / "output"

    model_directory.mkdir()

    pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "crop_line_id": ["line1"],
            "environment_id": ["env1"],
            "feature_score": [0.8],
        }
    ).to_parquet(input_path, index=False)

    (model_directory / "feature_manifest.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing model file",
    ):
        run_batch_ranking(
            input_path=input_path,
            model_directory=model_directory,
            output_directory=output_directory,
            model_package_version=2,
            model_package_arn="arn:test",
            model_data_uri="s3://test/model.tar.gz",
            input_data_uri="s3://test/input.parquet",
            ranking_columns=[
                "crop_line_id",
                "environment_id",
            ],
        )


def test_run_batch_ranking_raises_when_manifest_is_missing(tmp_path):
    import joblib

    from gene_edit_ranking.inference.batch_rank import run_batch_ranking

    input_path = tmp_path / "input.parquet"
    model_directory = tmp_path / "model"
    output_directory = tmp_path / "output"

    model_directory.mkdir()

    pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "crop_line_id": ["line1"],
            "environment_id": ["env1"],
            "feature_score": [0.8],
        }
    ).to_parquet(input_path, index=False)

    joblib.dump(
        PersistedFakeModel(),
        model_directory / "model.joblib",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing feature manifest",
    ):
        run_batch_ranking(
            input_path=input_path,
            model_directory=model_directory,
            output_directory=output_directory,
            model_package_version=2,
            model_package_arn="arn:test",
            model_data_uri="s3://test/model.tar.gz",
            input_data_uri="s3://test/input.parquet",
            ranking_columns=[
                "crop_line_id",
                "environment_id",
            ],
        )
