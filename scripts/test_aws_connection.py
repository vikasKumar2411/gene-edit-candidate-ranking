"""Verify Python connectivity to AWS STS and the configured S3 bucket."""

import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gene_edit_ranking.config import load_config


def main() -> None:
    """Validate the AWS identity and expected S3 bucket."""

    config = load_config()

    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]

    session = boto3.Session(region_name=region)

    try:
        sts_client = session.client("sts")
        identity = sts_client.get_caller_identity()

        print("AWS STS connection: SUCCESS")
        print(f"Account: {identity['Account']}")
        print(f"ARN: {identity['Arn']}")
        print(f"Region: {session.region_name}")

        s3_client = session.client("s3")
        s3_client.head_bucket(Bucket=bucket)

        print("S3 bucket access: SUCCESS")
        print(f"Bucket: {bucket}")

    except NoCredentialsError as exc:
        raise RuntimeError(
            "Python could not find AWS credentials."
        ) from exc
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(exc))

        raise RuntimeError(
            f"AWS request failed [{code}]: {message}"
        ) from exc
    except BotoCoreError as exc:
        raise RuntimeError(f"AWS SDK error: {exc}") from exc


if __name__ == "__main__":
    main()
