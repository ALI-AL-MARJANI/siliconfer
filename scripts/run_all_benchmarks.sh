#!/bin/bash
# Speed benchmarks and per-claim evaluations, one after another.
# Everything here is timing-sensitive or memory-hungry: run it on an otherwise
# idle machine, with nothing else using the GPU. See RUNBOOK.md for durations.
#
# Usage (from the repo root):  bash scripts/run_all_benchmarks.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"

step() { echo; echo "=== $* ==="; date -u +"%Y-%m-%dT%H:%M:%SZ"; "$@"; echo "--- exit $? ---"; }

# -- speed (results/bench/) ---------------------------------------------------
step "$PYTHON" scripts/bench_gemv.py
step "$PYTHON" scripts/bench_decode.py --model_id "$MODEL"
# External reference, in a throwaway environment (mlx-lm is not a dependency here).
if command -v uvx >/dev/null; then
  step uvx --from mlx-lm --with datasets python scripts/bench_mlx_lm_reference.py --model_id "$MODEL"
else
  echo "uvx not found: skipping the mlx-lm reference (results/bench/decode_mlx_lm_reference_*.json)"
fi
step "$PYTHON" scripts/bench_metal_attention.py
step "$PYTHON" scripts/bench_ttft.py --model_id "$MODEL"

# -- per-claim evaluations (results/claims/) ----------------------------------
step "$PYTHON" scripts/eval_kv_cache.py --model_id "$MODEL"
step "$PYTHON" scripts/eval_speculative.py --model_id "$MODEL"
# Mixed precision (research experiment, fake-quant only): demote the 5 least
# sensitive of 24 blocks to 2 bits, then to 3 bits. HQQ at both bit-widths.
step "$PYTHON" scripts/quantize.py --model_id "$MODEL" --method mixed --mixed_low_bits 2 \
  --mixed_budget_ratio 0.9 --mixed_permutations 4 --max_tokens 32768 \
  --results_json results/claims/mixed_precision_2bit.json
step "$PYTHON" scripts/quantize.py --model_id "$MODEL" --method mixed --mixed_low_bits 3 \
  --mixed_budget_ratio 0.948 --mixed_permutations 4 --max_tokens 32768 \
  --results_json results/claims/mixed_precision_3bit.json
step "$PYTHON" scripts/train_draft_head.py --model_id "$MODEL"
echo "=== ALL DONE ==="
