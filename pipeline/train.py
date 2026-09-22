"""Train on the packed blocks, with a held-out split, correct token ids, and checkpoints that survive.

Differences from the run that produced the published model:
  - every position of the block is trained, not just the first 64 tokens of each line
  - documents are separated by <|endoftext|>, so the model learns to stop
  - eos/bos/pad in the config are the ids this tokenizer really uses (the published config says 50256,
    which does not exist in a 33,977-token vocabulary)
  - a held-out set is scored while training, so overfitting is visible instead of hidden behind a training
    loss that fell to 0.019
  - the learning rate suits training from scratch; 5e-5 is a fine-tuning rate
  - it checkpoints, because a long run on interruptible hardware without checkpoints is a run you lose

    python3 pipeline/train.py --data corpus/packed --out runs/base --epochs 4 \
        --batch-size 8 --grad-accum 6 --lr 6e-4 --checkpoint-every 1000 --max-hours 17

--resume picks up from the checkpoint folder; --init continues from published weights instead.
The loop is plain PyTorch on purpose: every part of it is visible, and it runs the same on a Mac (MPS) and
on a GPU (CUDA, bfloat16).
"""
import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def load_blocks(path, block):
    """Memory-mapped uint16 blocks. At 466M tokens, materialising int64 would cost 3.7 GB of RAM."""
    ids = np.memmap(path, dtype=np.uint16, mode="r")
    count = len(ids) // block
    return ids[: count * block].reshape(count, block)


def take(blocks, rows):
    return torch.from_numpy(np.asarray(blocks[rows], dtype=np.int64))


def learning_rate(step, total, peak, warmup, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def evaluate(model, blocks, device, batch_size, autocast, limit):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, min(blocks.shape[0], limit), batch_size):
        chunk = take(blocks, slice(i, i + batch_size)).to(device)
        with autocast():
            total += model(chunk, labels=chunk).loss.item() * chunk.shape[0]
        count += chunk.shape[0]
    model.train()
    return total / max(count, 1)


def save(model, tok, folder, progress):
    os.makedirs(folder, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    tok.save_pretrained(folder)
    with open(os.path.join(folder, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="corpus/packed", help="folder written by pack_corpus.py")
    p.add_argument("--out", default="runs/base")
    p.add_argument("--init", default=None, help="start from these weights instead of a fresh model")
    p.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint if it exists")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--batch-size", type=int, default=8, help="blocks per forward pass")
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=700)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0, help="0 for a real corpus; 0.1 helps on a tiny one")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--embd", type=int, default=768)
    p.add_argument("--eval-every", type=int, default=500, help="optimiser steps between held-out checks")
    p.add_argument("--eval-blocks", type=int, default=400, help="held-out blocks per check")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--max-hours", type=float, default=0, help="stop and save when the budget is spent")
    p.add_argument("--patience", type=int, default=0, help="checks without improvement before stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu, mps or cuda; by default the fastest available")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    block = manifest["block"]
    train_blocks = load_blocks(os.path.join(args.data, "train.bin"), block)
    val_blocks = load_blocks(os.path.join(args.data, "val.bin"), block)
    tok = AutoTokenizer.from_pretrained(manifest["tokenizer"])
    eos, pad = manifest["eos_token_id"], tok.convert_tokens_to_ids("<|pad|>")

    checkpoint = os.path.join(args.out, "checkpoint")
    start_step, history = 0, []
    if args.resume and os.path.exists(os.path.join(checkpoint, "progress.json")):
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
        with open(os.path.join(checkpoint, "progress.json"), encoding="utf-8") as f:
            saved = json.load(f)
        start_step, history = saved.get("step", 0), saved.get("history", [])
        print(f"resuming from step {start_step:,}", flush=True)
    elif args.init:
        model = GPT2LMHeadModel.from_pretrained(args.init)
        model.config.update({"eos_token_id": eos, "bos_token_id": eos, "pad_token_id": pad})
    else:
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tok), n_positions=block, n_ctx=block, n_embd=args.embd, n_layer=args.layers,
            n_head=args.heads, bos_token_id=eos, eos_token_id=eos, pad_token_id=pad,
            resid_pdrop=args.dropout, embd_pdrop=args.dropout, attn_pdrop=args.dropout))
    model.generation_config.update(eos_token_id=eos, bos_token_id=eos, pad_token_id=pad)

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device).train()
    use_bf16 = device == "cuda"
    # nullcontext, not enable_grad: enable_grad inside evaluate()'s no_grad would rebuild the whole graph
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext

    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    per_step = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, train_blocks.shape[0] // per_step)
    total_steps = int(args.epochs * steps_per_epoch)
    params = sum(q.numel() for q in model.parameters())
    unique_tokens = train_blocks.shape[0] * block
    print(f"{params/1e6:.1f}M parameters | block {block} | {train_blocks.shape[0]:,} train blocks, "
          f"{val_blocks.shape[0]:,} held out | {device}{' bf16' if use_bf16 else ' fp32'}", flush=True)
    print(f"{per_step * block:,} tokens per step | {total_steps:,} steps = "
          f"{total_steps * per_step * block/1e9:.2f}B tokens over {args.epochs:g} passes of "
          f"{unique_tokens/1e6:,.0f}M", flush=True)
    if unique_tokens * args.epochs < 20 * params:
        print(f"note: {20*params/1e9:.1f}B tokens would be compute-optimal for {params/1e6:.0f}M parameters; "
              f"this run sees {total_steps*per_step*block/1e9:.2f}B.", flush=True)

    best, best_state, waited = float("inf"), None, 0
    order = torch.randperm(train_blocks.shape[0])
    cursor, started = 0, time.time()
    stop_reason = "finished the schedule"
    for step in range(start_step, total_steps):
        for group in opt.param_groups:
            group["lr"] = learning_rate(step, total_steps, args.lr, args.warmup)
        opt.zero_grad(set_to_none=True)
        losses = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.batch_size > order.shape[0]:
                order, cursor = torch.randperm(train_blocks.shape[0]), 0
            batch = take(train_blocks, order[cursor:cursor + args.batch_size].numpy()).to(device)
            cursor += args.batch_size
            with autocast():
                loss = model(batch, labels=batch).loss / args.grad_accum
            loss.backward()
            losses += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(losses) or not torch.isfinite(norm):
            print(f"step {step+1}: loss {losses} and gradient norm {norm} are not finite. Nothing past this "
                  f"point can recover, so the run stops here rather than burning the rest of the schedule.\n"
                  f"On Apple silicon this is usually memory pressure: try a smaller --batch-size, a smaller "
                  f"model, or --device cpu.", flush=True)
            stop_reason = "loss stopped being finite"
            break
        opt.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            val = evaluate(model, val_blocks, device, args.batch_size, autocast, args.eval_blocks)
            seen = (step + 1 - start_step) * per_step * block
            rate = seen / (time.time() - started)
            left = (total_steps - step - 1) * per_step * block / max(rate, 1) / 3600
            print(f"step {step+1:6,}/{total_steps:,}  train {losses:6.3f}  held-out {val:6.3f} "
                  f"(perplexity {math.exp(min(val, 20)):8,.1f})  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{rate:7,.0f} tokens/s  {left:4.1f}h left", flush=True)
            history.append({"step": step + 1, "train_loss": losses, "eval_loss": val,
                            "tokens_seen": seen, "tokens_per_second": round(rate)})
            if val < best - 1e-4:
                best, waited = val, 0
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if args.patience and waited >= args.patience:
                    stop_reason = f"held-out loss flat for {waited} checks"
                    print(stop_reason + "; stopping", flush=True)
                    break

        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            save(model, tok, checkpoint, {"step": step + 1, "best_eval_loss": best, "history": history})

        if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
            stop_reason = f"reached the {args.max_hours:g}h budget at step {step+1:,}"
            print(stop_reason, flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    model.to("cpu").save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"parameters": params, "block": block, "lr": args.lr, "epochs": args.epochs,
                   "stopped_because": stop_reason, "unique_train_tokens": unique_tokens,
                   "val_tokens": val_blocks.shape[0] * block, "best_eval_loss": best,
                   "best_eval_perplexity": math.exp(min(best, 20)), "history": history}, f, indent=1)
    print(f"\n{stop_reason} | best held-out loss {best:.3f} "
          f"(perplexity {math.exp(min(best, 20)):,.1f}) | saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
