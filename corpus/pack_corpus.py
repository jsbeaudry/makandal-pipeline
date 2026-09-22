"""Tokenise the built corpus and pack it into training blocks, streaming so memory stays flat.

`pipeline/prepare_data.py` holds every token in a Python list, which is fine for a 900k-token corpus and
impossible for a 300M-token one — a list of 300M Python ints is several gigabytes. This writes the token
ids straight to disk in uint16 batches instead, and never holds more than one batch of documents.

    python3 corpus/pack_corpus.py --corpus corpus/data --tokenizer corpus/tokenizer --out corpus/packed

The held-out split is chosen by a hash of the document, so it is stable across runs and no document can
land in both splits. Documents are separated by <|endoftext|>, which is what teaches the model to stop.

Writes train.bin, val.bin (uint16) and manifest.json, which is what pipeline/train.py reads.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import time

import numpy as np
from transformers import AutoTokenizer


def documents(folder, exclude=()):
    for path in sorted(glob.glob(os.path.join(folder, "shard-*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("text") and record.get("source", "") not in exclude:
                    yield record["text"], record.get("source", "")


def batches(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Writer:
    """Appends token ids to a .bin file, dropping only the final partial block."""

    def __init__(self, path, block):
        self.file = open(path, "wb")
        self.block = block
        self.spill = np.empty(0, dtype=np.uint16)
        self.tokens = self.blocks = 0

    def add(self, ids):
        data = np.concatenate([self.spill, np.asarray(ids, dtype=np.uint16)])
        whole = len(data) // self.block * self.block
        if whole:
            data[:whole].tofile(self.file)
            self.blocks += whole // self.block
            self.tokens += whole
        self.spill = data[whole:]

    def close(self):
        self.file.close()
        return {"tokens": self.tokens, "blocks": self.blocks, "tokens_dropped_at_the_end": int(self.spill.size)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/data")
    p.add_argument("--tokenizer", default="corpus/tokenizer")
    p.add_argument("--out", default="corpus/packed")
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--val-share", type=float, default=0.005, help="a small share is plenty at this size")
    p.add_argument("--batch-docs", type=int, default=1000)
    p.add_argument("--limit-docs", type=int, default=0)
    p.add_argument("--exclude-sources", default="", help="comma separated source names to leave out")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    if len(tok) > 65535:
        raise SystemExit(f"a {len(tok):,}-token vocabulary does not fit in uint16; change the dtype")
    os.makedirs(args.out, exist_ok=True)
    writers = {"train": Writer(os.path.join(args.out, "train.bin"), args.block),
               "val": Writer(os.path.join(args.out, "val.bin"), args.block)}

    cut = int(args.val_share * 2**32)
    counts, sources, started, seen = {"train": 0, "val": 0}, {}, time.time(), 0
    exclude = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    if exclude:
        print(f"leaving out: {', '.join(sorted(exclude))}", flush=True)
    stream = documents(args.corpus, exclude)
    for batch in batches(stream, args.batch_docs):
        texts = [t for t, _ in batch]
        encoded = tok(texts)["input_ids"]
        for (text, source), ids in zip(batch, encoded):
            # a hash of the document decides the split, so it is stable and never lands in both
            where = "val" if int(hashlib.blake2b(text.encode(), digest_size=4).hexdigest(), 16) < cut else "train"
            writers[where].add(ids + [eos])
            counts[where] += 1
            sources[source] = sources.get(source, 0) + 1
        seen += len(batch)
        if seen % 50_000 == 0:
            rate = seen / max(time.time() - started, 1)
            done = writers["train"].tokens + writers["val"].tokens
            print(f"  {seen:,} documents, {done/1e6:,.0f}M tokens, {rate:,.0f} docs/s", flush=True)
        if args.limit_docs and seen >= args.limit_docs:
            break

    manifest = {"tokenizer": os.path.abspath(args.tokenizer), "block": args.block, "eos_token_id": eos,
                "vocab_size": len(tok), "documents": counts, "sources": sources,
                "splits": {name: w.close() for name, w in writers.items()},
                "minutes": round((time.time() - started) / 60, 1)}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    for name, split in manifest["splits"].items():
        print(f"{name:5s}: {counts[name]:9,} documents, {split['tokens']:12,} tokens "
              f"-> {split['blocks']:9,} blocks of {args.block}")
    total = sum(s["tokens"] for s in manifest["splits"].values())
    print(f"\n{total:,} tokens packed in {manifest['minutes']} min -> {args.out}")
    print(f"a 112M-parameter model wants about 2.2B tokens, so this is {total/2.24e9:.0%} of one pass")


if __name__ == "__main__":
    main()
