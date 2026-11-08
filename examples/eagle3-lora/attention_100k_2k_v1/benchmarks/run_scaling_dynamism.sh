#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec python "$repo_root/scripts/benchmark_attention_scaling.py" "$@"
