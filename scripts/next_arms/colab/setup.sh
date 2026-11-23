#!/usr/bin/env bash
# One-time VM setup: unpack the uploaded repo snapshot and install it with vLLM.
# Idempotent; writes /content/.setup_done on success.
set -euo pipefail
[ -f /content/.setup_done ] && exit 0
mkdir -p /content/repo
tar -xzf /content/repo.tar.gz -C /content/repo
cd /content/repo
python -m pip install -q -e ./hs_connectors
python -m pip install -q -e '.[generator]' 'vllm==0.31.0'
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "vllm", vllm.__version__)
PY
touch /content/.setup_done
