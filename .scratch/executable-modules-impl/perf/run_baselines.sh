#!/usr/bin/env bash
# Decode-performance baseline runner (executable-modules-impl ticket 01).
#
# Method note: the gates draft names benchmark_decode.py, but that script drives
# the pytorch backend (lmdeploy.pytorch.decode.Engine) and is blind to the
# TurboMind engine this effort refactors. The baseline is therefore recorded
# with benchmark/profile_throughput.py --backend turbomind, the repo's
# fixed-concurrency decode benchmark that drives TurboMind directly.
#
# Shapes (identical for every model): random prompts, input_len=128,
# output_len=256, seed 0, tp=1, batch sizes 1 / 32 / 128.
# Load time: fresh process timing TurboMind.from_pretrained with the bs=128
# engine config. Raw logs and CSVs land in perf/raw/.
set -uo pipefail

cd /data/lmdeploy-gemm
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/mnt_cfs/huggingface_hub/hub/
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PYTHONPATH=/data/lmdeploy-gemm
# offline runs; the SOCKS ALL_PROXY breaks httpx at import
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
DATASET=/mnt_cfs/ShareGPT_V3_unfiltered_cleaned_split.json
OUT=.scratch/executable-modules-impl/perf/raw
mkdir -p "$OUT"

# entries: model|tag|model-format (empty model-format = default hf)
models=(
  "Qwen/Qwen3-8B|dense-bf16|"
  "Qwen/Qwen3-8B-AWQ|dense-awq-w4a16|awq"
  "Qwen/Qwen3-30B-A3B-Thinking-2507|moe-bf16|"
  "Qwen/Qwen3.5-35B-A3B-FP8|moe-fp8|fp8"
)

# optional args: run only the given tags
for entry in "${models[@]}"; do
  IFS='|' read -r model tag fmt <<<"$entry"
  if [ "$#" -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$tag"; then
    continue
  fi
  echo "=== $(date -Is) model=${model} tag=${tag} ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/gpu_usage.log"

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
      --csv "$OUT/${tag}_bs${bs}.csv" 2>&1 | tee -a "$OUT/${tag}_bs${bs}.log"
  done
done
echo "=== ALL RUNS DONE $(date -Is) ==="
