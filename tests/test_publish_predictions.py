"""Tests for prediction-publication utilities."""

from __future__ import annotations

import json

import pytest

from gene_edit_ranking.inference.publish_predictions import (
    parse_s3_uri,
    sha256_file,
    sha256_run_metadata,
)


def test_parse_s3_uri_returns_bucket_and_prefix():
    bucket, prefix = parse_s3_uri(
        "s3://gene-edit-ranking-bucket/gold/predictions/run-1/"
    )

    assert bucket == "gene-edit-ranking-bucket"
    assert prefix == "gold/predictions/run-1"


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "https://example.com/file",
        "gene-edit-ranking-bucket/path",
        "s3:///missing-bucket",
    ],
)
def test_parse_s3_uri_rejects_invalid_values(invalid_uri):
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        parse_s3_uri(invalid_uri)


def test_sha256_file_is_stable_for_identical_content(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_text("same content", encoding="utf-8")
    second.write_text("same content", encoding="utf-8")

    assert sha256_file(first) == sha256_file(second)


def test_metadata_checksum_ignores_generation_timestamp(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    common_metadata = {
        "input_rows": 2500,
        "output_rows": 2253,
        "model_package_version": 2,
        "selected_threshold": 0.35,
    }

    first.write_text(
        json.dumps(
            {
                **common_metadata,
                "generated_at_utc": "2026-07-26T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    second.write_text(
        json.dumps(
            {
                **common_metadata,
                "generated_at_utc": "2026-07-26T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert sha256_run_metadata(first) == sha256_run_metadata(second)


def test_metadata_checksum_changes_for_semantic_difference(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first.write_text(
        json.dumps(
            {
                "input_rows": 2500,
                "output_rows": 2253,
                "generated_at_utc": "2026-07-26T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    second.write_text(
        json.dumps(
            {
                "input_rows": 2500,
                "output_rows": 2200,
                "generated_at_utc": "2026-07-26T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert sha256_run_metadata(first) != sha256_run_metadata(second)


class FakeClientError(Exception):
    """Minimal stand-in for botocore ClientError."""

    def __init__(self, code: str):
        self.response = {
            "Error": {
                "Code": code,
            }
        }


class FakeS3:
    """Small fake S3 client for publication tests."""

    class exceptions:
        ClientError = FakeClientError

    def __init__(self, existing_checksum: str | None = None):
        self.existing_checksum = existing_checksum
        self.upload_calls = []

    def head_object(self, *, Bucket: str, Key: str):
        if self.existing_checksum is None:
            raise FakeClientError("404")

        return {
            "Metadata": {
                "sha256": self.existing_checksum,
            }
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict,
    ):
        self.upload_calls.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "extra_args": ExtraArgs,
            }
        )


def test_uploads_when_object_does_not_exist(tmp_path):
    from gene_edit_ranking.inference.publish_predictions import (
        upload_if_unchanged_or_missing,
    )

    local_file = tmp_path / "output.json"
    local_file.write_text("content", encoding="utf-8")

    s3 = FakeS3(existing_checksum=None)

    status = upload_if_unchanged_or_missing(
        s3=s3,
        local_path=local_file,
        bucket="test-bucket",
        key="predictions/output.json",
        content_type="application/json",
        checksum="abc123",
    )

    assert status == "uploaded"
    assert len(s3.upload_calls) == 1
    assert s3.upload_calls[0]["bucket"] == "test-bucket"
    assert s3.upload_calls[0]["key"] == "predictions/output.json"
    assert s3.upload_calls[0]["extra_args"]["Metadata"]["sha256"] == "abc123"


def test_skips_upload_when_existing_checksum_matches(tmp_path):
    from gene_edit_ranking.inference.publish_predictions import (
        upload_if_unchanged_or_missing,
    )

    local_file = tmp_path / "output.json"
    local_file.write_text("content", encoding="utf-8")

    s3 = FakeS3(existing_checksum="abc123")

    status = upload_if_unchanged_or_missing(
        s3=s3,
        local_path=local_file,
        bucket="test-bucket",
        key="predictions/output.json",
        content_type="application/json",
        checksum="abc123",
    )

    assert status == "already_exists_identical"
    assert s3.upload_calls == []


def test_refuses_overwrite_when_existing_checksum_differs(tmp_path):
    from gene_edit_ranking.inference.publish_predictions import (
        upload_if_unchanged_or_missing,
    )

    local_file = tmp_path / "output.json"
    local_file.write_text("content", encoding="utf-8")

    s3 = FakeS3(existing_checksum="different-checksum")

    with pytest.raises(
        RuntimeError,
        match="Refusing to overwrite an existing S3 object",
    ):
        upload_if_unchanged_or_missing(
            s3=s3,
            local_path=local_file,
            bucket="test-bucket",
            key="predictions/output.json",
            content_type="application/json",
            checksum="abc123",
        )

    assert s3.upload_calls == []
