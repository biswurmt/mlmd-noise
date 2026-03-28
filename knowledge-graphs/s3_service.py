"""s3_service.py
================
Nebius S3-compatible object storage helpers for the knowledge graph pipeline.

Required env vars (when S3_BUCKET is set):
  S3_ENDPOINT_URL        — e.g. https://storage.eu-north1.nebius.cloud
  S3_ACCESS_KEY_ID       — Nebius object storage access key
  S3_SECRET_ACCESS_KEY   — Nebius object storage secret key
  S3_BUCKET              — bucket name (e.g. diagnotix-kg)

Optional:
  S3_REGION              — region name (default: eu-north1)

Bucket layout:
  <bucket>/
    triage_knowledge_graph.pkl          ← latest PKL (overwritten each build)
    guideline_rules.json                ← latest rules (overwritten each build)
    backups/
      triage_knowledge_graph_<ts>.pkl   ← timestamped backup per build
"""

import datetime
import os


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "eu-north1"),
    )


def upload_build_artifacts(pkl_path: str, rules_path: str) -> None:
    """Upload the PKL and rules JSON to S3 after each graph build.

    Creates a timestamped backup of the PKL in backups/ and also
    overwrites the "latest" keys so the next cold start gets fresh data.
    Logs a warning and continues if S3 credentials are invalid — a failed
    upload should never block the graph build.
    """
    bucket = os.environ["S3_BUCKET"]
    try:
        s3 = _client()
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Versioned PKL backup
        s3.upload_file(pkl_path, bucket, f"backups/triage_knowledge_graph_{ts}.pkl")
        print(f"[s3] Backed up PKL → s3://{bucket}/backups/triage_knowledge_graph_{ts}.pkl")

        # Overwrite "latest" copies
        s3.upload_file(pkl_path, bucket, "triage_knowledge_graph.pkl")
        s3.upload_file(rules_path, bucket, "guideline_rules.json")
        print(f"[s3] Updated latest artifacts in s3://{bucket}/")
    except Exception as exc:
        print(f"[s3] WARNING: upload failed (check S3 credentials) — {exc}")


def sync_from_s3_if_missing(local_pkl: str, local_rules: str) -> None:
    """Download PKL and rules JSON from S3 if the local copies are absent.

    Called on FastAPI startup so a fresh container on Nebius automatically
    recovers the latest graph without a manual copy step.
    """
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return

    s3 = _client()
    for local_path, s3_key in [
        (local_pkl, "triage_knowledge_graph.pkl"),
        (local_rules, "guideline_rules.json"),
    ]:
        if not os.path.exists(local_path):
            try:
                s3.download_file(bucket, s3_key, local_path)
                print(f"[s3] Downloaded s3://{bucket}/{s3_key} → {local_path}")
            except Exception as exc:
                print(f"[s3] Could not download {s3_key}: {exc}")
