"""Instruction-tune the base model, with the loss on the response only.

The point of prompt masking: if the loss covers the instruction too, the model spends most of its
capacity learning to generate plausible instructions, which nobody will ever ask it to do. Here the
prompt's label positions are -100, so gradients come only from the response and its <|endoftext|>.
On this dataset that is 48.5% of the tokens — half the sequence is context, not target.

Shape of the data, measured with the real tokenizer over all 30,000 pairs rather than assumed:

    prompt         median  37   p90  59   p99  73   max 119
    response+eos   median  36   p90  54   p99  65   max  99
    total          median  76   p90  96   p99 115   max 158

Two things follow. Nothing needs truncating — the longest example is 158 tokens against a 1,024-token
block — so the failure that produced the first Makandal cannot happen here, and the script refuses to
truncate rather than doing it quietly if a future dataset is longer. And padding every batch to a fixed
width would waste 85% of the compute, so batches are length-bucketed instead: sequences of similar
length ride together and padding costs almost nothing. Packing several examples into one block would be
faster still, but GPT-2 attention has no block-diagonal mask, so example B would attend to example A.
A run this short does not need to buy speed with a correctness compromise.

    python3 pipeline/train_sft.py --data sft-30k.jsonl --base jsbeaudry/makandal-base \
        --out runs/sft --epochs 3 --batch-size 16 --grad-accum 4 --lr 5e-5

There is no --resume: 30,000 examples at ~76 tokens is 2.3M tokens per epoch, minutes of GPU time, so
a run that dies is cheaper to restart than to reconstruct.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import time

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

PROMPT = "### Enstriksyon:\n%s\n\n### Repons:\n"

# Plain text, not new special tokens. Added tokens arrive with random embeddings that 30,000 examples
# would have to train from scratch; these are pieces the model already saw 466M tokens of.


def held_out(instruction, share):
    """Deterministic split on the instruction, so the same pair lands in the same side every run."""
    digest = hashlib.blake2b(instruction.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") < share * 2 ** 32


def encode(tok, rows, max_len, eos):
    """Tokenise into ids and labels, with the prompt masked out of the loss."""
    examples, skipped = [], {"prompt too long": 0, "response would be cut": 0}
    for row in rows:
        prompt = tok(PROMPT % row["enstriksyon"], add_special_tokens=False)["input_ids"]
        answer = tok(row["repons"], add_special_tokens=False)["input_ids"] + [eos]
        if len(prompt) + 8 > max_len:
            skipped["prompt too long"] += 1
            continue
        if len(prompt) + len(answer) > max_len:
            # Cutting the tail and appending <|endoftext|> would teach the model to stop mid-sentence.
            # That is the same class of silent corruption that made the first model never stop at all,
            # so the example is dropped and counted instead.
            skipped["response would be cut"] += 1
            continue
        examples.append({"ids": prompt + answer,
                         "labels": [-100] * len(prompt) + answer,
                         "supervised": len(answer)})
    return examples, skipped


def build_batches(lengths, batch_size, generator, bucket=8):
    """Length-bucketed batches, reshuffled each epoch so the same examples do not always ride together."""
    order = torch.randperm(len(lengths), generator=generator).tolist()
    order.sort(key=lambda i: lengths[i] // bucket)        # stable, so within-bucket order stays random
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    return [chunks[i] for i in torch.randperm(len(chunks), generator=generator).tolist()]


def collate(items, pad_id, device):
    width = max(len(item["ids"]) for item in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for row, item in enumerate(items):
        n = len(item["ids"])
        ids[row, :n] = torch.tensor(item["ids"], dtype=torch.long)
        labels[row, :n] = torch.tensor(item["labels"], dtype=torch.long)
        mask[row, :n] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def learning_rate(step, total, peak, warmup, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def evaluate(model, examples, chunks, pad_id, device, autocast):
    """Held-out loss and next-token accuracy, over response tokens only."""
    model.eval()
    loss_sum, tokens, correct = 0.0, 0, 0
    for chunk in chunks:
        ids, labels, mask = collate([examples[i] for i in chunk], pad_id, device)
        with autocast():
            out = model(ids, attention_mask=mask, labels=labels)
        target = labels[:, 1:]                             # the model predicts position i+1 from i
        supervised = target != -100
        count = int(supervised.sum())
        if not count:
            continue
        loss_sum += out.loss.item() * count                # HF averages per batch; reweight by tokens
        tokens += count
        correct += int((out.logits[:, :-1].argmax(-1)[supervised] == target[supervised]).sum())
    model.train()
    return loss_sum / max(tokens, 1), correct / max(tokens, 1)


@torch.no_grad()
def show(model, tok, instructions, device, new_tokens=80):
    model.eval()
    for instruction in instructions:
        batch = tok(PROMPT % instruction, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**batch, max_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        answer = tok.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    Q  {instruction[:76]}", flush=True)
        print(f"    A  {' '.join(answer.split())[:190]}", flush=True)
    model.train()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="sft-30k.jsonl", help="jsonl with enstriksyon/repons fields")
    p.add_argument("--base", default="jsbeaudry/makandal-base", help="weights to tune")
    p.add_argument("--out", default="runs/sft")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16, help="examples per forward pass")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="a fine-tuning rate; 6e-4 is for from-scratch")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=0.0, help="the base model already generalises")
    p.add_argument("--max-len", type=int, default=256, help="longest example in the 30k set is 158")
    p.add_argument("--val-share", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=50, help="optimiser steps between held-out checks")
    p.add_argument("--sample-every", type=int, default=200, help="0 to never generate during training")
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu, mps or cuda; by default the fastest available")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    tok = AutoTokenizer.from_pretrained(args.base)
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    val_rows = [r for r in rows if held_out(r["enstriksyon"], args.val_share)]
    train_rows = [r for r in rows if not held_out(r["enstriksyon"], args.val_share)]
    train, skipped = encode(tok, train_rows, args.max_len, eos)
    val, _ = encode(tok, val_rows, args.max_len, eos)
    if not train or not val:
        raise SystemExit("nothing left to train on after encoding; check --max-len and the data file")

    supervised = sum(e["supervised"] for e in train)
    total_tokens = sum(len(e["ids"]) for e in train)
    assert min(e["supervised"] for e in train) > 0, "an example with no supervised token got through"
    print(f"{len(rows):,} pairs -> {len(train):,} train, {len(val):,} held out", flush=True)
    if sum(skipped.values()):
        print(f"skipped: {json.dumps(skipped)}", flush=True)
    print(f"{supervised:,} supervised of {total_tokens:,} tokens "
          f"({supervised / total_tokens:.1%}); longest example {max(len(e['ids']) for e in train)} tokens",
          flush=True)

    model = GPT2LMHeadModel.from_pretrained(args.base)
    model.config.update({"eos_token_id": eos, "bos_token_id": eos, "pad_token_id": pad})
    model.generation_config.update(eos_token_id=eos, bos_token_id=eos, pad_token_id=pad)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device).train()
    use_bf16 = device == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext

    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    lengths = [len(e["ids"]) for e in train]
    steps_per_epoch = max(1, len(build_batches(lengths, args.batch_size, generator)) // args.grad_accum)
    total_steps = max(1, int(args.epochs * steps_per_epoch))
    val_chunks = build_batches([len(e["ids"]) for e in val], args.batch_size, generator)
    probes = [val[0], val[len(val) // 2]] and [val_rows[0]["enstriksyon"],
                                               val_rows[len(val_rows) // 2]["enstriksyon"]]
    params = sum(q.numel() for q in model.parameters())
    print(f"{params / 1e6:.1f}M parameters | {device}{' bf16' if use_bf16 else ' fp32'} | "
          f"{args.batch_size * args.grad_accum} examples per step | {total_steps:,} steps "
          f"over {args.epochs:g} epochs", flush=True)

    start_loss, start_accuracy = evaluate(model, val, val_chunks, pad, device, autocast)
    print(f"before tuning: held-out response loss {start_loss:.3f} "
          f"(perplexity {math.exp(min(start_loss, 20)):,.1f}), next-token accuracy "
          f"{start_accuracy:.1%}", flush=True)

    best, best_state, history = float("inf"), None, []
    step, done, started = 0, False, time.time()
    stop_reason = "finished the schedule"
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        chunks = build_batches(lengths, args.batch_size, generator)
        for i in range(0, len(chunks) - args.grad_accum + 1, args.grad_accum):
            for group in opt.param_groups:
                group["lr"] = learning_rate(step, total_steps, args.lr, args.warmup)
            opt.zero_grad(set_to_none=True)
            losses = 0.0
            for chunk in chunks[i:i + args.grad_accum]:
                ids, labels, mask = collate([train[j] for j in chunk], pad, device)
                with autocast():
                    loss = model(ids, attention_mask=mask, labels=labels).loss / args.grad_accum
                loss.backward()
                losses += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(losses) or not torch.isfinite(norm):
                stop_reason = f"loss {losses} and gradient norm {norm} stopped being finite"
                print(f"step {step + 1}: {stop_reason}; stopping rather than burning the schedule",
                      flush=True)
                done = True
                break
            opt.step()
            step += 1

            if step % args.eval_every == 0 or step == total_steps:
                val_loss, accuracy = evaluate(model, val, val_chunks, pad, device, autocast)
                elapsed = time.time() - started
                left = (total_steps - step) * elapsed / max(step, 1) / 60
                print(f"step {step:5,}/{total_steps:,}  epoch {epoch + 1}  train {losses:6.3f}  "
                      f"held-out {val_loss:6.3f}  accuracy {accuracy:5.1%}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {left:4.1f} min left", flush=True)
                history.append({"step": step, "train_loss": losses, "eval_loss": val_loss,
                                "response_accuracy": accuracy})
                if val_loss < best - 1e-4:
                    best = val_loss
                    best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

            if args.sample_every and step % args.sample_every == 0:
                show(model, tok, probes, device)

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                model.save_pretrained(os.path.join(args.out, "checkpoint"), safe_serialization=True)

            if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
                stop_reason = f"reached the {args.max_hours:g}h budget at step {step:,}"
                print(stop_reason, flush=True)
                done = True
                break
            if step >= total_steps:
                done = True
                break

    if best_state:
        model.load_state_dict(best_state)
    final_loss, final_accuracy = evaluate(model, val, val_chunks, pad, device, autocast)

    os.makedirs(args.out, exist_ok=True)
    model.to("cpu").save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    summary = {"base": args.base, "pairs": len(rows), "train": len(train), "held_out": len(val),
               "skipped": skipped, "supervised_token_share": round(supervised / total_tokens, 4),
               "prompt_template": PROMPT, "epochs": args.epochs, "lr": args.lr, "steps": step,
               "stopped_because": stop_reason, "minutes": round((time.time() - started) / 60, 1),
               "before": {"eval_loss": start_loss, "response_accuracy": start_accuracy},
               "after": {"eval_loss": final_loss, "response_accuracy": final_accuracy},
               "history": history}
    with open(os.path.join(args.out, "sft_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print(f"\n{stop_reason}", flush=True)
    print(f"held-out response loss {start_loss:.3f} -> {final_loss:.3f}  |  "
          f"next-token accuracy {start_accuracy:.1%} -> {final_accuracy:.1%}  |  saved to {args.out}",
          flush=True)
    if args.sample_every:
        print("\nfinal samples:", flush=True)
        model.to(device)
        show(model, tok, probes, device, new_tokens=120)


if __name__ == "__main__":
    main()
