"""Score a model the way the training loss cannot: on text it never saw, and on context it never used.

Five checks:

  perplexity        does it predict Kreyòl held out from training, and Kreyòl from elsewhere?
  by position       fed a document from its first token, where does prediction break down?
  context control   is a 64-token window from mid-document as good as one from a document start?
                    if it is much worse, only document prefixes were trained
  memorisation      given the first 32 tokens of a training document, does it recite the rest?
  weights           how many of the model's position rows are still at their initial values?
                    a position never in a training sequence never gets a gradient

    python3 pipeline/evaluate.py --model pipeline/runs/proof --data pipeline/prepared
    python3 pipeline/evaluate.py --model jsbeaudry/makandal-pre-trained --data pipeline/prepared

`--other file.txt` scores any other Kreyòl file, which is how you see whether the model learned the language
or just this corpus.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def blocks(path, block):
    ids = np.memmap(path, dtype=np.uint16, mode="r")
    return torch.from_numpy(ids[: len(ids) // block * block].astype(np.int64)).view(-1, block)


def perplexity(model, batch, device):
    total = 0.0
    with torch.no_grad():
        for i in range(batch.shape[0]):
            chunk = batch[i:i + 1].to(device)
            total += model(chunk, labels=chunk).loss.item()
    return math.exp(min(total / max(batch.shape[0], 1), 20))


def by_position(model, docs, device, block, bands):
    """Loss and next-token accuracy per position, with every document aligned to its own first token."""
    loss_sum, hit_sum, count = torch.zeros(block - 1), torch.zeros(block - 1), 0
    for ids in docs:
        if ids.shape[0] < block:
            continue
        ids = ids[:block].to(device)
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0].float().cpu()
        target = ids[1:].cpu()
        loss_sum += torch.nn.functional.cross_entropy(logits[:-1], target, reduction="none")
        hit_sum += (logits[:-1].argmax(-1) == target).float()
        count += 1
    if not count:
        return []
    return [(a, b, (loss_sum[a:b] / count).mean().item(), (hit_sum[a:b] / count).mean().item())
            for a, b in bands if b <= block - 1]


def window_loss(model, docs, device, start, length=64):
    """Loss on a standalone `length`-token window taken from `start` in each document."""
    total, count = 0.0, 0
    for ids in docs:
        if ids.shape[0] < start + length:
            continue
        window = ids[start:start + length].unsqueeze(0).to(device)
        with torch.no_grad():
            total += model(window, labels=window).loss.item()
        count += 1
    return (total / count if count else float("nan")), count


def memorisation(model, tok, docs, device, prompt_tokens=32, new_tokens=64):
    """Share of the model's continuation that is a word-for-word copy of the real one."""
    scores = []
    for ids in docs:
        if ids.shape[0] < prompt_tokens + new_tokens:
            continue
        prompt = ids[:prompt_tokens].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=new_tokens, do_sample=False,
                                 pad_token_id=model.config.pad_token_id or 0)
        made = tok.decode(out[0][prompt_tokens:], skip_special_tokens=True).split()
        real = tok.decode(ids[prompt_tokens:prompt_tokens + new_tokens], skip_special_tokens=True).split()
        if not made or not real:
            continue
        run = 0
        for m, r in zip(made, real):                      # how far the copy runs before it diverges
            if m != r:
                break
            run += 1
        scores.append((run, sum(m == r for m, r in zip(made, real)) / len(made)))
    if not scores:
        return None
    return {"documents": len(scores),
            "words_copied_before_diverging": sum(s[0] for s in scores) / len(scores),
            "share_word_for_word": sum(s[1] for s in scores) / len(scores)}


def position_profile(model, window=16):
    """A position never present in a training sequence never gets a gradient, so its row keeps the spread
    the initialiser gave it. Rows are averaged in windows because a single row is too noisy to judge."""
    wpe = model.transformer.wpe.weight.detach().float().cpu()
    spread = wpe.std(dim=1)
    means = spread[: spread.shape[0] // window * window].view(-1, window).mean(dim=1)
    init = getattr(model.config, "initializer_range", 0.02)
    tail = means[means.shape[0] // 2:]
    if (tail.mean() - init).abs() > 0.05 * init:          # the far context has moved: all of it trained
        return wpe.shape[0], wpe.shape[0], means, init
    spread_of_tail = (tail - tail.median()).abs().median().clamp(min=1e-6)
    moved = (means - tail.median()).abs() > 6 * spread_of_tail
    trained = (int(moved.nonzero().max().item()) + 1) * window if moved.any() else 0
    return trained, wpe.shape[0], means, init


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="pipeline/prepared")
    p.add_argument("--other", nargs="*", default=[], help="more Kreyòl text files to score")
    p.add_argument("--train-docs", default="data/*.txt", help="documents to test position and memorisation")
    p.add_argument("--docs", type=int, default=12)
    p.add_argument("--prompts", nargs="*", default=["Ayiti se yon peyi", "Lekòl la", "Matematik se"])
    p.add_argument("--max-blocks", type=int, default=40)
    p.add_argument("--device", default=None, help="cpu, mps or cuda; by default the fastest available")
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    block = min(manifest["block"], model.config.n_positions)
    print(f"{args.model}  ({sum(q.numel() for q in model.parameters())/1e6:.1f}M parameters, "
          f"context {model.config.n_positions}, vocabulary {model.config.vocab_size})")
    print(f"eos={model.config.eos_token_id} bos={model.config.bos_token_id} pad={model.config.pad_token_id} "
          f"| the tokenizer's eos is {tok.eos_token_id}"
          f"{'  <- cannot be emitted' if (model.config.eos_token_id or 0) >= model.config.vocab_size else ''}")

    trained, total, means, init = position_profile(model)
    print(f"\nposition rows that ever received a gradient: the first {trained:,} of {total:,}"
          f"  (spread per 16 rows, initialiser {init})")
    step = max(1, means.shape[0] // 8)
    print("  " + "  ".join(f"{i*16}:{means[i]:.4f}" for i in range(0, means.shape[0], step)))
    if trained < total:
        print(f"  rows {trained}-{total-1} are untouched, so anything past token {trained} is noise")

    val = blocks(os.path.join(args.data, "val.bin"), block)[:args.max_blocks]
    train = blocks(os.path.join(args.data, "train.bin"), block)[:args.max_blocks]
    print(f"\nperplexity on {block}-token blocks (an untrained model scores {model.config.vocab_size:,})")
    print(f"  held-out documents      {perplexity(model, val, device):12,.1f}")
    print(f"  training documents      {perplexity(model, train, device):12,.1f}")
    for path in args.other:
        text = open(os.path.expanduser(path), encoding="utf-8", errors="replace").read()
        ids = tok(text[:400_000], return_tensors="pt")["input_ids"][0]
        other = ids[: len(ids) // block * block].view(-1, block)[:args.max_blocks]
        if other.shape[0]:
            print(f"  {os.path.basename(path)[:22]:22s}  {perplexity(model, other, device):12,.1f}")

    docs = [tok(open(f, encoding="utf-8", errors="replace").read(), return_tensors="pt")["input_ids"][0]
            for f in sorted(glob.glob(args.train_docs))[:args.docs]]
    bands = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, block - 1)]
    rows = by_position(model, docs, device, block, bands)
    if rows:
        print("\nloss by position, each document fed from its first token (flat means the context is used)")
        print(f"  {'positions':>13}  {'loss':>6}  {'perplexity':>11}  {'next token right':>16}")
        for a, b, loss, hit in rows:
            print(f"  {a:5d}-{b:<7d}  {loss:6.3f}  {math.exp(min(loss, 20)):11,.1f}  {100*hit:15.1f}%")

    start_loss, n = window_loss(model, docs, device, 0)
    mid_loss, _ = window_loss(model, docs, device, 600)
    print(f"\n64-token window, {n} documents (a gap means only document beginnings were trained)")
    print(f"  from the document start   loss {start_loss:6.3f}   perplexity {math.exp(min(start_loss,20)):10,.1f}")
    print(f"  from the middle (token 600) loss {mid_loss:6.3f}   perplexity {math.exp(min(mid_loss,20)):10,.1f}")

    score = memorisation(model, tok, docs, device)
    if score:
        print(f"\nmemorisation, given the first 32 tokens of {score['documents']} training documents")
        print(f"  words recited before diverging    {score['words_copied_before_diverging']:6.1f}")
        print(f"  share of the continuation copied  {100*score['share_word_for_word']:5.1f}%")

    print("\nsamples")
    for prompt in args.prompts:
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.9, top_p=0.9,
                                 repetition_penalty=1.1, pad_token_id=model.config.pad_token_id or 0,
                                 eos_token_id=tok.eos_token_id)
        stopped = out[0][-1].item() == tok.eos_token_id
        print(f"  {prompt!r} -> {tok.decode(out[0], skip_special_tokens=True)[:300]}"
              f"{'  [stopped on its own]' if stopped else ''}")


if __name__ == "__main__":
    main()
