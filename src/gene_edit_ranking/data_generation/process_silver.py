"""Transform validated Raw CSV datasets into typed Silver Parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config
from gene_edit_ranking.validation.silver_schemas import (
    DatasetSchema,
    SILVER_SCHEMAS,
)


DATASET_FILES = {
    "genes": "genes.csv",
    "crop_lines": "crop_lines.csv",
    "environments": "environments.csv",
    "candidates": "candidate_edits.csv",
    "experiments": "experiments.csv",
}


@dataclass(frozen=True)
class ProcessingSummary:
    """Summary of one Raw-to-Silver dataset transformation."""

    dataset_name: str
    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    output_path: Path


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize common boolean representations into pandas bool values."""

    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    result = normalized.map(mapping)

    if result.isna().any():
        invalid_values = sorted(series[result.isna()].astype(str).unique())
        raise ValueError(
            f"Unrecognized boolean values: {invalid_values[:10]}"
        )

    return result.astype(bool)


def apply_schema(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> pd.DataFrame:
    """Apply the configured Silver data types to a dataframe."""

    result = dataframe.copy()

    expected_columns = (
        schema.string_columns
        + schema.integer_columns
        + schema.float_columns
        + schema.boolean_columns
        + schema.date_columns
    )

    missing_columns = set(expected_columns) - set(result.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = result.loc[:, list(expected_columns)]

    for column in schema.string_columns:
        result[column] = result[column].astype("string").str.strip()

    for column in schema.integer_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("int64")

    for column in schema.float_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("float64")

    for column in schema.boolean_columns:
        result[column] = normalize_boolean(result[column])

    for column in schema.date_columns:
        result[column] = pd.to_datetime(
            result[column],
            errors="raise",
            utc=True,
        )

    return result


def add_lineage_columns(
    dataframe: pd.DataFrame,
    source_file: str,
    source_version: str,
    processing_date: str,
) -> pd.DataFrame:
    """Add operational lineage columns to a Silver dataframe."""

    result = dataframe.copy()

    result["_source_file"] = source_file
    result["_source_version"] = source_version
    result["_processing_date"] = pd.to_datetime(
        processing_date,
        utc=True,
    )
    result["_processed_at_utc"] = datetime.now(timezone.utc)

    return result


def process_dataset(
    dataset_name: str,
    raw_path: Path,
    output_path: Path,
    schema: DatasetSchema,
    source_version: str,
    processing_date: str,
    compression: str,
    add_lineage: bool,
) -> ProcessingSummary:
    """Transform one Raw CSV file into a Silver Parquet file."""

    raw_dataframe = pd.read_csv(raw_path)
    input_rows = len(raw_dataframe)

    silver_dataframe = apply_schema(raw_dataframe, schema)

    duplicate_mask = silver_dataframe.duplicated(
        subset=[schema.primary_key],
        keep="first",
    )
    duplicate_rows_removed = int(duplicate_mask.sum())

    silver_dataframe = silver_dataframe.loc[
        ~duplicate_mask
    ].reset_index(drop=True)

    if silver_dataframe[schema.primary_key].isna().any():
        raise ValueError(
            f"{dataset_name} contains null primary-key values."
        )

    if add_lineage:
        silver_dataframe = add_lineage_columns(
            dataframe=silver_dataframe,
            source_file=raw_path.name,
            source_version=source_version,
            processing_date=processing_date,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    silver_dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    return ProcessingSummary(
        dataset_name=dataset_name,
        input_rows=input_rows,
        output_rows=len(silver_dataframe),
        duplicate_rows_removed=duplicate_rows_removed,
        output_path=output_path,
    )


def process_all(config: dict[str, Any]) -> list[ProcessingSummary]:
    """Process all configured datasets into the local Silver layer."""

    raw_dir = PROJECT_ROOT / config["data"]["local_raw_dir"]
    silver_dir = PROJECT_ROOT / "data" / "silver"

    source_version = config["data"]["source_version"]
    silver_config = config["silver_processing"]

    processing_date = silver_config["processing_date"]
    compression = silver_config["compression"]
    add_lineage = silver_config["add_lineage_columns"]

    summaries: list[ProcessingSummary] = []

    for dataset_name, filename in DATASET_FILES.items():
        raw_path = raw_dir / filename

        output_path = (
            silver_dir
            / dataset_name
            / f"processing_date={processing_date}"
            / f"source_version={source_version}"
            / f"{dataset_name}.parquet"
        )

        summary = process_dataset(
            dataset_name=dataset_name,
            raw_path=raw_path,
            output_path=output_path,
            schema=SILVER_SCHEMAS[dataset_name],
            source_version=source_version,
            processing_date=processing_date,
            compression=compression,
            add_lineage=add_lineage,
        )

        summaries.append(summary)

    return summaries


def main() -> None:
    """Run local Raw-to-Silver processing."""

    config = load_config()
    summaries = process_all(config)

    print("Raw-to-Silver processing: SUCCESS")
    print()

    for summary in summaries:
        print(
            f"{summary.dataset_name:<12} "
            f"input_rows={summary.input_rows:>5} "
            f"output_rows={summary.output_rows:>5} "
            f"duplicates_removed={summary.duplicate_rows_removed:>3} "
            f"file={summary.output_path}"
        )


if __name__ == "__main__":
    main()
