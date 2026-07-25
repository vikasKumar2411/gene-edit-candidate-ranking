"""Generate referentially consistent synthetic datasets for local development."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gene_edit_ranking.config import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class DatasetBundle:
    """Container for the five generated Raw datasets."""

    genes: pd.DataFrame
    crop_lines: pd.DataFrame
    environments: pd.DataFrame
    candidates: pd.DataFrame
    experiments: pd.DataFrame


def _create_ids(prefix: str, count: int, width: int = 5) -> list[str]:
    """Create stable identifiers such as GENE-00001."""

    return [f"{prefix}-{index:0{width}d}" for index in range(1, count + 1)]


def generate_genes(
    rng: np.random.Generator,
    count: int,
) -> pd.DataFrame:
    """Generate synthetic gene metadata."""

    gene_ids = _create_ids("GENE", count)

    chromosomes = rng.choice(
        ["CHR01", "CHR02", "CHR03", "CHR04", "CHR05", "CHR06"],
        size=count,
    )

    start_positions = rng.integers(10_000, 90_000_000, size=count)
    gene_lengths = rng.integers(800, 25_000, size=count)

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": [f"SYN_GENE_{index:04d}" for index in range(1, count + 1)],
            "chromosome": chromosomes,
            "start_position": start_positions,
            "end_position": start_positions + gene_lengths,
            "gene_family": rng.choice(
                [
                    "transcription_factor",
                    "transport_protein",
                    "signaling_protein",
                    "metabolic_enzyme",
                    "stress_response",
                ],
                size=count,
            ),
            "synthetic_conservation_score": np.round(
                rng.beta(3, 2, size=count),
                4,
            ),
            "is_synthetic": True,
        }
    )


def generate_crop_lines(
    rng: np.random.Generator,
    count: int,
) -> pd.DataFrame:
    """Generate synthetic crop-line metadata."""

    crop_line_ids = _create_ids("LINE", count)

    return pd.DataFrame(
        {
            "crop_line_id": crop_line_ids,
            "crop_type": rng.choice(
                ["maize", "soybean"],
                size=count,
                p=[0.65, 0.35],
            ),
            "breeding_program": rng.choice(
                ["program_alpha", "program_beta", "program_gamma"],
                size=count,
            ),
            "maturity_group": rng.choice(
                ["early", "mid", "late"],
                size=count,
            ),
            "baseline_yield_index": np.round(
                rng.normal(loc=100, scale=8, size=count).clip(75, 125),
                3,
            ),
            "baseline_drought_tolerance": np.round(
                rng.beta(2.5, 2.5, size=count),
                4,
            ),
            "is_synthetic": True,
        }
    )


def generate_environments(
    rng: np.random.Generator,
    count: int,
) -> pd.DataFrame:
    """Generate synthetic experimental environments."""

    environment_ids = _create_ids("ENV", count, width=3)

    rainfall = rng.normal(loc=480, scale=150, size=count).clip(120, 850)
    average_temperature = rng.normal(loc=24, scale=4, size=count).clip(14, 34)

    drought_index = (
        1
        - (rainfall - rainfall.min())
        / max(float(rainfall.max() - rainfall.min()), 1.0)
    )

    return pd.DataFrame(
        {
            "environment_id": environment_ids,
            "location_code": [f"SYN_LOC_{index:03d}" for index in range(1, count + 1)],
            "season": rng.choice(
                ["spring", "summer", "fall"],
                size=count,
            ),
            "soil_type": rng.choice(
                ["loam", "sandy_loam", "clay_loam", "silt_loam"],
                size=count,
            ),
            "rainfall_mm": np.round(rainfall, 2),
            "average_temperature_c": np.round(average_temperature, 2),
            "synthetic_drought_index": np.round(drought_index, 4),
            "is_synthetic": True,
        }
    )


def generate_candidates(
    rng: np.random.Generator,
    count: int,
    genes: pd.DataFrame,
    crop_lines: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic candidate edits."""

    candidate_ids = _create_ids("CAND", count, width=6)

    editing_methods = rng.choice(
        ["knockout", "base_edit", "promoter_edit"],
        size=count,
        p=[0.40, 0.35, 0.25],
    )

    predicted_edit_efficiency = rng.beta(4, 2, size=count)
    predicted_off_target_risk = rng.beta(1.8, 5, size=count)

    return pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "gene_id": rng.choice(genes["gene_id"], size=count),
            "crop_line_id": rng.choice(
                crop_lines["crop_line_id"],
                size=count,
            ),
            "editing_method": editing_methods,
            "target_position": rng.integers(1, 20_000, size=count),
            "predicted_edit_efficiency": np.round(
                predicted_edit_efficiency,
                4,
            ),
            "predicted_off_target_risk": np.round(
                predicted_off_target_risk,
                4,
            ),
            "design_batch": rng.choice(
                ["design_batch_001", "design_batch_002", "design_batch_003"],
                size=count,
            ),
            "created_date": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(
                rng.integers(0, 180, size=count),
                unit="D",
            ),
            "is_synthetic": True,
        }
    )


def generate_experiments(
    rng: np.random.Generator,
    count: int,
    candidates: pd.DataFrame,
    genes: pd.DataFrame,
    crop_lines: pd.DataFrame,
    environments: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic experimental outcomes with learnable signal."""

    experiment_ids = _create_ids("EXP", count, width=7)

    candidate_sample = candidates.sample(
        n=count,
        replace=True,
        random_state=101,
    ).reset_index(drop=True)

    experiment_data = pd.DataFrame(
        {
            "experiment_id": experiment_ids,
            "candidate_id": candidate_sample["candidate_id"],
            "environment_id": rng.choice(
                environments["environment_id"],
                size=count,
            ),
            "replicate_number": rng.integers(1, 5, size=count),
        }
    )

    enriched = (
        experiment_data
        .merge(
            candidates[
                [
                    "candidate_id",
                    "gene_id",
                    "crop_line_id",
                    "predicted_edit_efficiency",
                    "predicted_off_target_risk",
                ]
            ],
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            genes[
                [
                    "gene_id",
                    "synthetic_conservation_score",
                ]
            ],
            on="gene_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            crop_lines[
                [
                    "crop_line_id",
                    "baseline_yield_index",
                    "baseline_drought_tolerance",
                ]
            ],
            on="crop_line_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            environments[
                [
                    "environment_id",
                    "synthetic_drought_index",
                ]
            ],
            on="environment_id",
            how="left",
            validate="many_to_one",
        )
    )

    noise = rng.normal(0, 3.5, size=count)

    yield_change = (
        8.0 * enriched["predicted_edit_efficiency"]
        - 5.5 * enriched["predicted_off_target_risk"]
        + 2.5 * enriched["synthetic_conservation_score"]
        + 3.0 * enriched["baseline_drought_tolerance"]
        - 2.0 * enriched["synthetic_drought_index"]
        + noise
    )

    drought_response = (
        0.35 * enriched["predicted_edit_efficiency"]
        - 0.25 * enriched["predicted_off_target_risk"]
        + 0.30 * enriched["baseline_drought_tolerance"]
        + 0.20 * enriched["synthetic_conservation_score"]
        + rng.normal(0, 0.10, size=count)
    ).clip(-0.5, 1.5)

    observed_edit_success = rng.binomial(
        n=1,
        p=(
            0.10
            + 0.80 * enriched["predicted_edit_efficiency"]
            - 0.20 * enriched["predicted_off_target_risk"]
        ).clip(0.02, 0.98),
    )

    result = experiment_data.copy()
    result["observed_edit_success"] = observed_edit_success
    result["yield_change_percent"] = np.round(yield_change, 4)
    result["drought_response_score"] = np.round(drought_response, 4)
    result["measurement_quality_score"] = np.round(
        rng.beta(6, 1.5, size=count),
        4,
    )
    result["experiment_date"] = (
        pd.Timestamp("2026-04-01")
        + pd.to_timedelta(
            rng.integers(0, 120, size=count),
            unit="D",
        )
    )
    result["is_synthetic"] = True

    return result


def generate_all(config: dict[str, Any]) -> DatasetBundle:
    """Generate all five datasets from configuration."""

    settings = config["synthetic_data"]
    row_counts = settings["row_counts"]

    rng = np.random.default_rng(settings["random_seed"])

    genes = generate_genes(rng, row_counts["genes"])
    crop_lines = generate_crop_lines(rng, row_counts["crop_lines"])
    environments = generate_environments(rng, row_counts["environments"])

    candidates = generate_candidates(
        rng=rng,
        count=row_counts["candidates"],
        genes=genes,
        crop_lines=crop_lines,
    )

    experiments = generate_experiments(
        rng=rng,
        count=row_counts["experiments"],
        candidates=candidates,
        genes=genes,
        crop_lines=crop_lines,
        environments=environments,
    )

    return DatasetBundle(
        genes=genes,
        crop_lines=crop_lines,
        environments=environments,
        candidates=candidates,
        experiments=experiments,
    )


def write_datasets(
    datasets: DatasetBundle,
    output_dir: Path,
) -> dict[str, Path]:
    """Write generated datasets as CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "genes": output_dir / "genes.csv",
        "crop_lines": output_dir / "crop_lines.csv",
        "environments": output_dir / "environments.csv",
        "candidates": output_dir / "candidate_edits.csv",
        "experiments": output_dir / "experiments.csv",
    }

    datasets.genes.to_csv(output_paths["genes"], index=False)
    datasets.crop_lines.to_csv(output_paths["crop_lines"], index=False)
    datasets.environments.to_csv(output_paths["environments"], index=False)
    datasets.candidates.to_csv(output_paths["candidates"], index=False)
    datasets.experiments.to_csv(output_paths["experiments"], index=False)

    return output_paths


def main() -> None:
    """Generate and write all configured synthetic datasets."""

    config = load_config()
    output_dir = PROJECT_ROOT / config["data"]["local_raw_dir"]

    datasets = generate_all(config)
    output_paths = write_datasets(datasets, output_dir)

    print("Synthetic Raw data generation: SUCCESS")
    print(f"Output directory: {output_dir}")
    print()

    for dataset_name, path in output_paths.items():
        dataframe = getattr(datasets, dataset_name)

        print(
            f"{dataset_name:<12} "
            f"rows={len(dataframe):>5} "
            f"columns={len(dataframe.columns):>2} "
            f"file={path.name}"
        )


if __name__ == "__main__":
    main()
