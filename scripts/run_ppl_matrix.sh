#!/bin/bash
# Standard-protocol perplexity matrix (docs/eval-protocol.md).
#
# {fp16, rtn, hqq, sinq} + {gptq, awq, awq with llm-awq options} x 3 calibration
# seeds (plus two single-seed AWQ ablations on WikiText-2), each on WikiText-2
# test and a fixed C4 validation subset, seq_len 2048.
#
#   default   quick tier: 32,768 scored tokens per run      -> results/ppl/
#   FULL=1    full tier: the whole WikiText-2 test split and
#             64 x 2048 C4 tokens                           -> results/ppl_full/
#
# Each tier also runs the asymmetric (zero-point) variants of rtn / gptq /
# awq / sinq, and MLX's own quantizer as an external reference.
#
# Runs are sequential (one model in memory at a time) and resumable: a run
# whose output JSON already exists is skipped, so delete a file to force that
# run again.
#
# Usage (from the repo root):
#   bash scripts/run_ppl_matrix.sh            # both datasets, quick tier
#   bash scripts/run_ppl_matrix.sh wikitext2  # one dataset
#   FULL=1 bash scripts/run_ppl_matrix.sh     # full tier
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
# Use the repo venv when present, so the script also works from a shell
# where it has not been activated.
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}"
DATASETS="${1:-wikitext2 c4}"
SEQ_LEN=2048
if [ "${FULL:-0}" = 1 ]; then
  OUT_DIR=results/ppl_full; WT2_ARGS="";  C4_WINDOWS=64
else
  OUT_DIR=results/ppl;      WT2_ARGS="--max_tokens $((16 * SEQ_LEN))"; C4_WINDOWS=16
fi

run() {
  local out="$OUT_DIR/$1"; shift
  if [ -f "$out" ]; then echo "=== skip (exists): $out ==="; return; fi
  echo "=== $* ==="; date -u +"%Y-%m-%dT%H:%M:%SZ"
  "$PYTHON" scripts/eval_ppl.py --model_id "$MODEL" --seq_len $SEQ_LEN --out_dir "$OUT_DIR" "$@"
  echo "--- done: $* ---"
}

for d in $DATASETS; do
  case $d in
    wikitext2) data_args="--dataset wikitext2 $WT2_ARGS" ;;
    c4)        data_args="--dataset c4 --c4_n_seqs $C4_WINDOWS --c4_seed 0" ;;
    *) echo "unknown dataset: $d" >&2; exit 1 ;;
  esac
  for m in fp16 rtn hqq sinq; do
    run "${m}_${d}_seq${SEQ_LEN}.json" --method $m $data_args
  done
  # External reference quantizer (MLX's own mx.quantize) on the same layers.
  run "mlx_native_${d}_seq${SEQ_LEN}.json" --method mlx_native $data_args
  run "mlx_native-g64_${d}_seq${SEQ_LEN}.json" --method mlx_native --group_size 64 $data_args
  for s in 0 1 2; do
    for m in gptq awq; do
      run "${m}_${d}_seq${SEQ_LEN}_seed${s}.json" --method $m --calib_seed $s $data_args
    done
  done
  # AWQ with the two llm-awq options (block-output loss + weight clipping) ...
  for s in 0 1 2; do
    run "awq-blockloss-clip_${d}_seq${SEQ_LEN}_seed${s}.json" \
      --method awq --awq_block_loss --awq_clip --calib_seed $s $data_args
  done
  # Asymmetric (zero-point) grids. HQQ is always asymmetric, so it has no extra row.
  run "rtn-asym_${d}_seq${SEQ_LEN}.json" --method rtn --asym $data_args
  run "sinq-asym_${d}_seq${SEQ_LEN}.json" --method sinq --asym $data_args
  for s in 0 1 2; do
    run "gptq-asym_${d}_seq${SEQ_LEN}_seed${s}.json" --method gptq --asym --calib_seed $s $data_args
    run "awq-blockloss-clip-asym_${d}_seq${SEQ_LEN}_seed${s}.json" \
      --method awq --awq_block_loss --awq_clip --asym --calib_seed $s $data_args
  done
  # ... and each option alone (ablation, one seed, WikiText-2 only).
  if [ "$d" = wikitext2 ]; then
    run "awq-blockloss_${d}_seq${SEQ_LEN}_seed0.json" --method awq --awq_block_loss --calib_seed 0 $data_args
    run "awq-clip_${d}_seq${SEQ_LEN}_seed0.json" --method awq --awq_clip --calib_seed 0 $data_args
  fi
done
echo "=== ALL DONE ==="
