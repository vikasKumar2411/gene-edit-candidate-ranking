"""SageMaker-compatible inference server for gene-edit ranking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, Response, request

MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
MODEL_PATH = MODEL_DIR / "model.joblib"
MANIFEST_PATH = MODEL_DIR / "feature_manifest.json"

app = Flask(__name__)

_model: Any | None = None
_manifest: dict[str, Any] | None = None


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the model and feature manifest once."""

    global _model, _manifest

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Feature manifest not found: {MANIFEST_PATH}"
            )

        _model = joblib.load(MODEL_PATH)
        _manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

    return _model, _manifest


@app.get("/ping")
def ping() -> Response:
    """SageMaker health check."""

    try:
        load_artifacts()
    except Exception as exc:
        return Response(
            response=json.dumps({"status": "unhealthy", "error": str(exc)}),
            status=503,
            mimetype="application/json",
        )

    return Response(
        response=json.dumps({"status": "healthy"}),
        status=200,
        mimetype="application/json",
    )


@app.post("/invocations")
def invocations() -> Response:
    """Score one or more candidate records."""

    try:
        model, manifest = load_artifacts()

        payload = request.get_json(force=True)

        if isinstance(payload, dict):
            records = payload.get("instances", [payload])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(
                "Request body must be an object or list of objects."
            )

        if not records:
            raise ValueError("No records supplied for inference.")

        dataframe = pd.DataFrame(records)
        probabilities = model.predict_proba(dataframe)[:, 1]

        threshold = float(manifest["selected_threshold"])
        predictions = (probabilities >= threshold).astype(int)

        response_payload = {
            "model_name": manifest.get(
                "model_name",
                "edit-success-logistic-regression",
            ),
            "model_version": manifest.get(
                "model_version",
                "v1",
            ),
            "selected_threshold": threshold,
            "predictions": [
                {
                    "edit_success_probability": float(probability),
                    "predicted_edit_success": int(prediction),
                }
                for probability, prediction in zip(
                    probabilities,
                    predictions,
                    strict=True,
                )
            ],
        }

        return Response(
            response=json.dumps(response_payload),
            status=200,
            mimetype="application/json",
        )

    except Exception as exc:
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=400,
            mimetype="application/json",
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
    )
