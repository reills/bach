#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v5.ace_step import (  # noqa: E402
    ACE_STEP_DEFAULT_BRANCH,
    ACE_STEP_DEFAULT_TAG,
    ACE_STEP_DEFAULT_REPO_DIR,
    ACE_STEP_REPO_URL,
    build_ace_step_setup_plan,
    setup_ace_step_repo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone/update ACE-Step 1.5 as a pinned external dependency.")
    parser.add_argument("--repo-dir", default=ACE_STEP_DEFAULT_REPO_DIR)
    parser.add_argument("--repo-url", default=ACE_STEP_REPO_URL)
    parser.add_argument("--branch", default=ACE_STEP_DEFAULT_BRANCH, help="Remote branch to track for normal git pull.")
    parser.add_argument("--recommended-tag", default=ACE_STEP_DEFAULT_TAG, help="Release tag recorded in manifests/docs.")
    parser.add_argument("--install", action="store_true", help="Run uv sync after checkout.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running git/uv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_ace_step_setup_plan(
        repo_dir=args.repo_dir,
        repo_url=args.repo_url,
        branch=args.branch,
        recommended_tag=args.recommended_tag,
        install=args.install,
    )
    result = setup_ace_step_repo(plan, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
