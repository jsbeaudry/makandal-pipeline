"""Correct the token ids on the published model, and optionally shrink it to bfloat16.

`config.json` on jsbeaudry/makandal-pre-trained declares eos_token_id and bos_token_id 50256, an id that
does not exist in its 33,977-token vocabulary, so the model can never emit it and generation only stops at
max_new_tokens. The tokenizer's real ids are <|endoftext|> 0 and <|pad|> 1.

    python3 pipeline/fix_published_config.py --out pipeline/fixed          # writes a corrected copy
    python3 pipeline/fix_published_config.py --out pipeline/fixed --bf16   # and halves the file size

It writes a local folder; pushing is a separate, deliberate step:

    huggingface-cli upload jsbeaudry/makandal-pre-trained pipeline/fixed .

Correct ids do not make the model good. It was trained on the first 64 tokens of each document, so anything
past roughly 60 tokens is noise; the model card should say so.
"""
import argparse

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="jsbeaudry/makandal-pre-trained")
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", action="store_true", help="save in bfloat16: 448 MB becomes 224 MB")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    pad = tok.convert_tokens_to_ids("<|pad|>")
    print(f"before: config eos={model.config.eos_token_id} bos={model.config.bos_token_id} "
          f"pad={model.config.pad_token_id}, vocabulary {model.config.vocab_size}")
    if model.config.eos_token_id is not None and model.config.eos_token_id < model.config.vocab_size:
        print("the ids already fit the vocabulary; nothing to correct")

    model.config.update({"eos_token_id": eos, "bos_token_id": eos, "pad_token_id": pad})
    model.generation_config.update(eos_token_id=eos, bos_token_id=eos, pad_token_id=pad)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6
    print(f"after : config eos={eos} bos={eos} pad={pad}")
    print(f"written to {args.out} ({size:.0f} MB)")
    print("\ncheck it with:")
    print(f"  python3 pipeline/evaluate.py --model {args.out} --data pipeline/prepared")


if __name__ == "__main__":
    main()
