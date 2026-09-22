#!/usr/bin/env bash
# Instruction-tune the base model on a pod, then push the result.
#
#   bash -c "curl -fsSL https://raw.githubusercontent.com/jsbeaudry/makandal-pipeline/main/pod/run_sft_training.sh | bash"
#
# Needs HF_TOKEN to read the private dataset and push the model. No vLLM here, so no CUDA 13
# requirement — any recent torch host will do. 30,000 pairs at ~76 tokens is 2.3M tokens per epoch,
# so this is minutes of GPU time, not hours.
set -u
BASE="${BASE:-jsbeaudry/makandal-base}"
DATASET="${DATASET:-jsbeaudry/kreyol-sft}"
FILE="${FILE:-sft-30k.jsonl}"
TARGET="${TARGET:-jsbeaudry/makandal-instruct}"
EPOCHS="${EPOCHS:-3}"
REPO="${REPO:-https://github.com/jsbeaudry/makandal-pipeline.git}"
cd /workspace || exit 1

echo "[pod] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no gpu')"
pip install -q --no-input 'transformers>=4.44' huggingface_hub hf_transfer 2>&1 | tail -2
python3 -c 'import torch; print("[pod] torch", torch.__version__, torch.cuda.get_device_name(0))' \
  || { echo '[pod] FATAL: cuda unusable on this host'; sleep infinity; }

rm -rf /workspace/repo
git clone --depth 1 "$REPO" /workspace/repo || { echo '[pod] FATAL: clone failed'; sleep infinity; }

python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$DATASET', '$FILE', repo_type='dataset', local_dir='/workspace/data'))
" || { echo '[pod] FATAL: dataset download failed'; sleep infinity; }

# The masking is invisible at runtime — a script that masks nothing still trains and still reports a
# falling loss. Prove it here, before spending the GPU.
python3 /workspace/repo/pipeline/test_train_sft.py --tokenizer "$BASE" \
  || { echo '[pod] FATAL: train_sft checks failed'; sleep infinity; }

python3 -u /workspace/repo/pipeline/train_sft.py \
  --data "/workspace/data/$FILE" --base "$BASE" --out /workspace/sft --epochs "$EPOCHS"

python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$TARGET', private=True, exist_ok=True)
api.upload_folder(folder_path='/workspace/sft', repo_id='$TARGET')
print('[pod] uploaded /workspace/sft to $TARGET', flush=True)
" || { echo '[pod] FATAL: upload failed'; sleep infinity; }

echo '[pod] ===== DONE ====='
sleep infinity
