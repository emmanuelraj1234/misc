#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download, HfApi, hf_hub_url
from urllib.parse import urlparse

def get_repo_type(repo_id: str) -> str | None:
    api = HfApi()
    for repo_type in ["dataset", "model", "space"]:
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return repo_type  # return the first one that works
        except Exception:
            continue
    return None  # not found - None is treated as "model" by snapshot_download()



def main():
    parser = argparse.ArgumentParser(
        description="Download a snapshot from Hugging Face Hub."
    )
    parser.add_argument(
        "repo_id",
        help="The repository ID on Hugging Face Hub (e.g., 'bert-base-uncased')."
    )
    parser.add_argument(
        "--revision",
        help="Optional revision (branch, tag, or commit hash). Default is the latest.",
        default=None
    )
    parser.add_argument(
        "--etag-timeout",
        type=int,
        default=86400,
        help="ETag timeout in seconds (default: 86400)."
    )

    args = parser.parse_args()
    repo_type = get_repo_type(args.repo_id)

    print("Using HfApi endpoint:", HfApi().endpoint)
    print(f"Inferred repo_type:\n{repo_type}")
    print(f"Will start downloading revision '{args.revision}' from this repo_id:\n{args.repo_id}\n")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=repo_type,
        etag_timeout=args.etag_timeout,
        revision=args.revision
    )

if __name__ == "__main__":
    main()
