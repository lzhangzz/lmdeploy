#!/usr/bin/env bash
# Stage-1 gate perf runner (ticket 04). Identical shapes to run_baselines.sh;
# results land in perf/stage1/ for comparison against baseline.md.
set -uo pipefail

cd /data/lmdeploy-gemm
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/mnt_cfs/huggingface_hub/hub/
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTHONPATH=/data/lmdeploy-gemm
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
DATASET=/mnt_cfs/ShareGPT_V3_unfiltered_cleaned_split.json
OUT=.scratch/executable-modules-impl/perf/stage1
mkdir -p "$OUT"

models=(
  "Qwen/Qwen3-8B|dense-bf16|"
  "Qwen/Qwen3-8B-AWQ|dense-awq-w4a16|awq"
  "Qwen/Qwen3-30B-A3B-Thinking-2507|moe-bf16|"
  "Qwen/Qwen3.5-35B-A3B-FP8|moe-fp8|fp8"
)

for entry in "${models[@]}"; do
  IFS='|' read -r model tag fmt <<<"$entry"
  if [ "$#" -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$tag"; then
    continue
  fi
  echo "=== $(date -Is) model=${model} tag=${tag} ==="

  echo "--- load time: ${tag} ---"
  MODEL="$model" python - <<'EOF' 2>&1 | tee -a "$OUT/${tag}_load.log"
import os, time
from lmdeploy.turbomind import TurboMind
from lmdeploy.messages import TurbomindEngineConfig
t0 = time.monotonic()
tm = TurboMind.from_pretrained(os.environ['MODEL'],
                               engine_config=TurbomindEngineConfig(tp=1, max_batch_size=128))
print('LOAD_SECONDS %.2f' % (time.monotonic() - t0))
EOF

  for bs in 1 32 128; do
    n=$(( bs * 4 ))
    [ "$bs" -eq 1 ] && n=16
    echo "--- throughput: ${tag} bs=${bs} num_prompts=${n} ---"
    fmt_args=()
    [ -n "$fmt" ] && fmt_args=(--model-format "$fmt")
    python benchmark/profile_throughput.py "$DATASET" "$model" \
      --backend turbomind --concurrency "$bs" \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$n" --seed 0 "${fmt_args[@]}" \
      --csv "$OUT/${tag}_bs${bs}.csv" 2>&1 | tee -a "$OUT/${tag}_bs${bs}.log" | grep -E "Throughput|token/s|Error|error" | tail -3
  done
done
echo "=== ALL RUNS DONE $(date -Is) ==="
