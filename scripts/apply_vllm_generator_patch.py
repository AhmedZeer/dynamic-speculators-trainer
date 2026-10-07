"""Check/apply/revert the pinned vLLM adapter patch; no native rebuild needed."""

import argparse
import json
from pathlib import Path

from speculators.generator.serving.patch import patch_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("check", "apply", "revert"):
        actions.add_argument(
            f"--{action}", action="store_const", const=action, dest="action"
        )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        help="Directory containing vllm/; defaults to installed package",
    )
    args = parser.parse_args()
    print(json.dumps(patch_sources(args.action, args.vllm_root), indent=2))


if __name__ == "__main__":
    main()
