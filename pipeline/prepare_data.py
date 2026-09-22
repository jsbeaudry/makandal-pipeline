"""Turn documents into packed training blocks, without throwing any of them away.

The published model was trained by tokenising each document with truncation, so only the first 64 tokens of
each of the 414 files were ever seen: 26,496 tokens, 3% of the corpus, memorised. Everything past token 64
scores ~9.3 loss, barely better than the 10.4 of an untrained model.

This does it the usual way instead: every document is tokenised whole, an end-of-text token is put between
documents, the stream is concatenated and cut into blocks of `--block` tokens. Nothing is truncated, every
position in the block is trained, and the model learns where a document ends.

    python3 pipeline/prepare_data.py --input "data/*.txt" --out prepared

Writes prepared/train.bin, prepared/val.bin (uint16 token ids) and prepared/manifest.json.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "jsbeaudry/makandal-pre-trained"


def read_documents(patterns, json_field):
    """One document per file, or per record for a .json/.jsonl list."""
    docs = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.expanduser(pattern))):
            if path.endswith((".json", ".jsonl")):
                with open(path, encoding="utf-8") as f:
                    records = json.load(f) if path.endswith(".json") else [json.loads(l) for l in f if l.strip()]
                for i, record in enumerate(records):
                    text = record.get(json_field, "") if isinstance(record, dict) else str(record)
                    docs.append((f"{os.path.basename(path)}#{i}", text))
            else:
                with open(path, encoding="utf-8", errors="replace") as f:
                    docs.append((os.path.basename(path), f.read()))
    return docs


def clean(text):
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def shingles(text, n=13):
    words = text.split()
    return {hash(tuple(words[i:i + n])) for i in range(0, max(1, len(words) - n), 3)}


def deduplicate(docs, threshold):
    """Drop identical documents, then near-identical ones (shared 13-word runs)."""
    kept, seen_hashes, kept_shingles, exact, near = [], set(), [], 0, 0
    for name, text in docs:
        digest = hashlib.md5(text.encode()).hexdigest()
        if digest in seen_hashes:
            exact += 1
            continue
        seen_hashes.add(digest)
        mine = shingles(text)
        if mine and any(len(mine & other) / len(mine | other) > threshold for other in kept_shingles):
            near += 1
            continue
        kept.append((name, text))
        kept_shingles.append(mine)
    return kept, exact, near


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="file patterns, e.g. 'data/*.txt' or a .json list")
    p.add_argument("--json-field", default="text", help="which field holds the text in a .json list")
    p.add_argument("--out", default="prepared")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--block", type=int, default=1024, help="tokens per training block; use the model's context")
    p.add_argument("--val-share", type=float, default=0.05, help="share of documents held out, never trained on")
    p.add_argument("--min-chars", type=int, default=200, help="drop documents shorter than this")
    p.add_argument("--dedup", type=float, default=0.8, help="drop a document sharing more than this with an earlier one")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    docs = [(name, clean(text)) for name, text in read_documents(args.input, args.json_field)]
    short = sum(1 for _, t in docs if len(t) < args.min_chars)
    docs = [(n, t) for n, t in docs if len(t) >= args.min_chars]
    docs, exact, near = deduplicate(docs, args.dedup)
    if not docs:
        sys.exit("no documents left after cleaning")

    random.Random(args.seed).shuffle(docs)
    n_val = max(1, round(len(docs) * args.val_share))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}      # held out by document, never by block

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    os.makedirs(args.out, exist_ok=True)
    manifest = {"tokenizer": args.tokenizer, "block": args.block, "eos_token_id": eos, "seed": args.seed,
                "documents": {k: len(v) for k, v in splits.items()},
                "dropped": {"too_short": short, "exact_duplicates": exact, "near_duplicates": near},
                "splits": {}}

    for split, items in splits.items():
        ids = []
        for _, text in items:
            ids.extend(tok(text)["input_ids"])
            ids.append(eos)                                    # this is what teaches the model to stop
        blocks = len(ids) // args.block
        if blocks == 0:
            sys.exit(f"{split}: only {len(ids)} tokens, less than one block of {args.block}")
        array = np.array(ids[: blocks * args.block], dtype=np.uint16)
        array.tofile(os.path.join(args.out, f"{split}.bin"))
        manifest["splits"][split] = {"documents": len(items), "tokens": len(ids), "blocks": blocks,
                                     "tokens_kept": int(array.size), "tokens_dropped_at_the_end": len(ids) - int(array.size)}
        print(f"{split:5s}: {len(items):5d} documents, {len(ids):9,d} tokens -> {blocks:6,d} blocks of {args.block}")

    if tok.vocab_size > 65535:
        sys.exit("this vocabulary needs uint32; change the dtype")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    kept = sum(s["tokens_kept"] for s in manifest["splits"].values())
    print(f"\ndropped: {short} too short, {exact} identical, {near} near-identical")
    print(f"{kept:,} tokens packed into blocks; nothing truncated (the old pipeline kept 26,496)")
    print(f"manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
