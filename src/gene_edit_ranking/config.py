"""Configuration utilities for the Gene Edit Candidate Ranking project."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/dev.yaml") -> dict[str, Any]:
    """Load environment variables and the project YAML configuration."""

    load_dotenv(PROJECT_ROOT / ".env")

    resolved_path = PROJECT_ROOT / config_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config
