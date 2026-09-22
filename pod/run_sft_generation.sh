#!/usr/bin/env bash
# Serve the teacher model and generate the Kreyol instruction set.
#
# A pod runs this with one short command, so no code travels through an API field:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/jsbeaudry/makandal-pipeline/main/pod/run_sft_generation.sh | bash"
#
# Needs an HF_TOKEN environment variable to push the result. Wants a CUDA 13 host: installing vLLM
# pulls torch built for cu130, which will not initialise on an older driver.
set -u
TARGET="${TARGET:-30000}"
WORKERS="${WORKERS:-24}"
TEACHER="${TEACHER:-google/gemma-4-26b-a4b-it}"
REPO="${REPO:-https://github.com/jsbeaudry/makandal-pipeline.git}"
cd /workspace || exit 1

echo "[pod] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no gpu')"
pip install -q --no-input vllm hf_transfer huggingface_hub 2>&1 | tail -2
python3 -c 'import torch; print("[pod] torch", torch.__version__, torch.cuda.get_device_name(0))' \
  || { echo '[pod] FATAL: cuda unusable on this host'; sleep infinity; }

rm -rf /workspace/repo
git clone --depth 1 "$REPO" /workspace/repo || { echo '[pod] FATAL: clone failed'; sleep infinity; }
python3 -c "
import ast
source = open('/workspace/repo/corpus/generate_sft.py', encoding='utf-8').read()
ast.parse(source)
assert 'òtograf' in source, 'accents missing from the clone'
print('[pod] generator parses, accents intact, %d bytes' % len(source))
" || { echo '[pod] FATAL: generator did not survive the clone'; sleep infinity; }

nohup vllm serve "$TEACHER" --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 --port 8000 > /workspace/vllm.log 2>&1 &

echo "[pod] waiting for the teacher to load"
UP=0
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then echo "[pod] SERVER UP after ${i}0s"; UP=1; break; fi
  if [ $((i % 6)) -eq 0 ]; then echo "[pod] loading ${i}0s"; fi
  sleep 10
done
[ "$UP" = "1" ] || { echo '[pod] FATAL: server never came up'; tail -45 /workspace/vllm.log; sleep infinity; }

python3 -u /workspace/repo/corpus/generate_sft.py \
  --target "$TARGET" --workers "$WORKERS" --model "$TEACHER" --out /workspace/sft.jsonl
python3 -u /workspace/repo/pod/report_sft.py /workspace/sft.jsonl
python3 -u /workspace/repo/pod/push_sft.py /workspace/sft.jsonl

echo '[pod] ===== DONE ====='
sleep infinity
