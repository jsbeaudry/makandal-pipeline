"""Compare two models fairly, on the same text, when they do not share a tokenizer.

Perplexity per token cannot be compared across tokenizers: a model whose tokenizer packs more characters
into each token is predicting fewer, harder tokens. Bits per character removes that, because the
denominator is the text itself rather than how it happens to be split.

    python3 pipeline/compare.py --models jsbeaudry/makandal-pre-trained jsbeaudry/makandal-base \
        --documents corpus/data --limit 200

Every document comes from the held-out split, so neither model trained on it: it is held out from the new
model's run, and the published model was trained on other text entirely.
"""
import argparse
import glob
import hashlib
import gzip
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def held_out(folder, limit, val_share=0.005, min_chars=1500):
    """Only the documents pack_corpus.py put in the val split.

    The shards hold train and val mixed together — the split is decided by a hash of the document, not by
    which shard it landed in. Reusing that exact rule is what keeps this an honest held-out measurement
    rather than a reading of the new model's own training data.
    """
    cut = int(val_share * 2**32)
    out, scanned = [], 0
    for path in sorted(glob.glob(os.path.join(folder, "shard-*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                scanned += 1
                if len(text) < min_chars:
                    continue
                if int(hashlib.blake2b(text.encode(), digest_size=4).hexdigest(), 16) < cut:
                    out.append(text)
                    if len(out) >= limit:
                        print(f"found {len(out)} held-out documents in {scanned:,} scanned")
                        return out
    print(f"found {len(out)} held-out documents in {scanned:,} scanned")
    return out


def bits_per_character(model, tok, texts, device, block):
    """Total negative log-likelihood over the text, divided by the characters it spells."""
    nats = characters = tokens = 0.0
    for text in texts:
        ids = tok(text, return_tensors="pt")["input_ids"][0][:block]
        if ids.shape[0] < 32:
            continue
        piece = tok.decode(ids, skip_special_tokens=True)   # score exactly the characters we fed in
        chunk = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss.item()
        nats += loss * (ids.shape[0] - 1)                   # the first token is not predicted
        tokens += ids.shape[0] - 1
        characters += len(piece)
    return {"bits_per_character": nats / math.log(2) / max(characters, 1),
            "perplexity_per_token": math.exp(min(nats / max(tokens, 1), 20)),
            "tokens_per_character": tokens / max(characters, 1)}


def by_position(model, tok, texts, device, block, bands):
    total, hits, count = torch.zeros(block - 1), torch.zeros(block - 1), 0
    for text in texts:
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        if ids.shape[0] < block:
            continue
        ids = ids[:block].to(device)
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0].float().cpu()
        target = ids[1:].cpu()
        total += torch.nn.functional.cross_entropy(logits[:-1], target, reduction="none")
        hits += (logits[:-1].argmax(-1) == target).float()
        count += 1
    if not count:
        return []
    return [(a, b, (total[a:b] / count).mean().item(), (hits[a:b] / count).mean().item())
            for a, b in bands if b <= block - 1]


def window(model, tok, texts, device, start, length=64):
    total, count = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        if ids.shape[0] < start + length:
            continue
        piece = ids[start:start + length].unsqueeze(0).to(device)
        with torch.no_grad():
            total += model(piece, labels=piece).loss.item()
        count += 1
    return total / count if count else float("nan")


def stops(model, tok, device, prompts, new_tokens=120):
    """How often the model ends a document by itself instead of running to the limit."""
    ended = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=True, temperature=0.8,
                                 top_p=0.92, repetition_penalty=1.1,
                                 pad_token_id=tok.pad_token_id or 1, eos_token_id=tok.eos_token_id)
        ended.append(out[0][-1].item() == tok.eos_token_id or out.shape[1] < ids["input_ids"].shape[1] + new_tokens)
    return sum(ended), len(ended)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--documents", default="corpus/data")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    texts = held_out(args.documents, args.limit)
    print(f"{len(texts)} held-out documents, {sum(len(t) for t in texts):,} characters\n")
    prompts = ["Ayiti se yon peyi", "Lekòl la", "Matematik se", "Nan maten an", "Istwa peyi a",
               "Manje kreyòl", "Yon jou", "Doktè a di", "Timoun yo", "Nan mache a"]
    findings = {}

    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(name, token=os.environ.get("HF_TOKEN"),
                                                     dtype=torch.float32).to(args.device).eval()
        block = min(1024, model.config.n_positions)
        score = bits_per_character(model, tok, texts, args.device, block)
        rows = by_position(model, tok, texts, args.device, block,
                           [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, block - 1)])
        start, middle = window(model, tok, texts, args.device, 0), window(model, tok, texts, args.device, 600)
        ended, tried = stops(model, tok, args.device, prompts)
        findings[name] = {**score, "by_position": rows, "window_start": start, "window_middle": middle,
                          "stopped": ended, "prompts": tried, "vocabulary": len(tok),
                          "parameters": sum(q.numel() for q in model.parameters())}

        print(f"=== {name} ({findings[name]['parameters']/1e6:.1f}M parameters, "
              f"vocabulary {len(tok):,}) ===")
        print(f"  bits per character          {score['bits_per_character']:8.3f}   <- comparable across tokenizers")
        print(f"  perplexity per token        {score['perplexity_per_token']:8.1f}")
        print(f"  tokens per character        {score['tokens_per_character']:8.3f}")
        print(f"  {'positions':>14}  {'loss':>7}  {'next token right':>16}")
        for a, b, loss, hit in rows:
            print(f"  {a:6d}-{b:<7d}  {loss:7.3f}  {100*hit:15.1f}%")
        print(f"  64-token window from the start    loss {start:6.3f}")
        print(f"  64-token window from token 600    loss {middle:6.3f}")
        print(f"  ended a document by itself        {ended} of {tried} prompts\n", flush=True)
        del model

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=1)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
