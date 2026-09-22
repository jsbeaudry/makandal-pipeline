"""Check the parts of train_sft.py that fail silently: masking, padding, batching, alignment.

Prompt masking is invisible at runtime. A script that masks nothing still trains, still reports a
falling loss, and still produces a model — it just spends half its gradient on learning to write
instructions. So the masking is asserted here rather than trusted, and the whole loop is run end to
end on a tiny random model before any of it reaches a rented GPU.

    python3 pipeline/test_train_sft.py --tokenizer corpus/tokenizer
"""
import argparse
import json
import os
import sys
import tempfile

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft                                                            # noqa: E402

PAIRS = [
    {"enstriksyon": "Ki sa ki bon pou sante w nan maten?", "repons": "Bwè dlo epi manje yon fwi."},
    {"enstriksyon": "Bay de etap pou plante mayi.", "repons": "Pare tè a. Mete grenn yo nan twou yo."},
    {"enstriksyon": "Reekri sa pi poli: 'Ban m liv la!'", "repons": "Èske w ka ban m liv la, silvouplè?"},
    {"enstriksyon": "Klasifye mesaj sa a: 'Mwen kontan anpil.'", "repons": "Mesaj sa a gen yon ton pozitif."},
]


def check_masking(tok):
    eos = tok.eos_token_id
    examples, skipped = train_sft.encode(tok, PAIRS, max_len=256, eos=eos)
    assert len(examples) == len(PAIRS), f"lost examples: {skipped}"

    for example, pair in zip(examples, PAIRS):
        ids, labels = example["ids"], example["labels"]
        assert len(ids) == len(labels)
        masked = [i for i, label in enumerate(labels) if label == -100]
        kept = [i for i, label in enumerate(labels) if label != -100]
        assert masked == list(range(len(masked))), "the masked span must be the prompt prefix"
        assert kept == list(range(len(masked), len(ids))), "the supervised span must be the suffix"
        for i in kept:
            assert labels[i] == ids[i], "a supervised label must equal the token it supervises"
        assert ids[-1] == eos, "every example must end with <|endoftext|> so the model learns to stop"
        assert tok.decode([ids[i] for i in kept[:-1]]) == pair["repons"], "supervised span != response"
        prompt = tok.decode([ids[i] for i in masked])
        assert prompt == train_sft.PROMPT % pair["enstriksyon"], "masked span != prompt"
    print(f"  masking       {len(examples)} examples: prompt masked, response supervised, ends with eos")


def check_refuses_to_truncate(tok):
    """A response that does not fit must be dropped, never cut and capped with eos."""
    long_pair = [{"enstriksyon": "Rakonte istwa a.", "repons": "Mwen te ale nan mache a. " * 200}]
    examples, skipped = train_sft.encode(tok, long_pair, max_len=64, eos=tok.eos_token_id)
    assert examples == [], "an over-long response was kept"
    assert skipped["response would be cut"] == 1, skipped

    wide_prompt = [{"enstriksyon": "Ki sa ki pase? " * 200, "repons": "Anyen."}]
    examples, skipped = train_sft.encode(tok, wide_prompt, max_len=64, eos=tok.eos_token_id)
    assert examples == [], "an over-long prompt was kept"
    assert skipped["prompt too long"] == 1, skipped
    print("  truncation    over-long prompts and responses are dropped and counted, never cut")


def check_collate(tok):
    examples, _ = train_sft.encode(tok, PAIRS, max_len=256, eos=tok.eos_token_id)
    pad = tok.pad_token_id
    ids, labels, mask = train_sft.collate(examples, pad, "cpu")
    width = max(len(e["ids"]) for e in examples)
    assert ids.shape == labels.shape == mask.shape == (len(examples), width)
    for row, example in enumerate(examples):
        n = len(example["ids"])
        assert mask[row, :n].all() and not mask[row, n:].any(), "attention mask must cover real tokens"
        assert (ids[row, n:] == pad).all(), "the tail must be padding"
        assert (labels[row, n:] == -100).all(), "padding must never be supervised"
        assert int((labels[row] != -100).sum()) == example["supervised"]
    print(f"  collate       padded to {width}, mask and labels agree on where the real tokens stop")


def check_batches():
    lengths = [7, 3, 9, 1, 5, 2, 8, 4, 6, 10, 11, 12]
    generator = torch.Generator().manual_seed(0)
    chunks = train_sft.build_batches(lengths, batch_size=4, generator=generator, bucket=4)
    seen = sorted(i for chunk in chunks for i in chunk)
    assert seen == list(range(len(lengths))), "every example must appear exactly once per epoch"
    padding = sum(max(lengths[i] for i in chunk) * len(chunk) - sum(lengths[i] for i in chunk)
                  for chunk in chunks)
    flat = sum(max(lengths) - length for length in lengths)
    assert padding < flat, f"bucketing did not reduce padding: {padding} vs {flat}"
    print(f"  batching      covers every example once; padding {padding} tokens against {flat} unbucketed")


def check_split():
    instructions = [f"kesyon nimewo {i}" for i in range(20000)]
    share = sum(train_sft.held_out(text, 0.02) for text in instructions) / len(instructions)
    assert 0.015 < share < 0.025, f"held-out share drifted to {share:.3f}"
    assert all(train_sft.held_out(t, 0.02) == train_sft.held_out(t, 0.02) for t in instructions[:100])
    print(f"  split         deterministic, {share:.2%} held out at --val-share 0.02")


def check_loss_ignores_prompt(tok):
    """The reported loss must equal cross-entropy over the response positions and nothing else.

    Asserting only that the masked and unmasked losses differ is too weak: on an untrained model every
    token costs about ln(vocab), so the two are close whether or not the mask works. This recomputes the
    loss by hand over the supervised positions and demands an exact match, which also pins down the
    off-by-one between logits and labels that evaluate() relies on.
    """
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=32, n_layer=2,
                                       n_head=2, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.pad_token_id)).eval()
    examples, _ = train_sft.encode(tok, PAIRS, max_len=256, eos=tok.eos_token_id)
    ids, labels, mask = train_sft.collate(examples, tok.pad_token_id, "cpu")
    with torch.no_grad():
        reported = model(ids, attention_mask=mask, labels=labels)
        whole = model(ids, attention_mask=mask, labels=ids.masked_fill(mask == 0, -100)).loss.item()
    target = labels[:, 1:]
    supervised = target != -100
    by_hand = torch.nn.functional.cross_entropy(reported.logits[:, :-1][supervised],
                                                target[supervised]).item()
    assert abs(by_hand - reported.loss.item()) < 1e-4, \
        f"loss {reported.loss.item():.6f} != response-only cross-entropy {by_hand:.6f}"
    # The shift drops labels[:, 0], which is always a prompt token, so it costs no supervision: every
    # response token is predicted, the first of them from the last token of the prompt.
    assert int(supervised.sum()) == sum(e["supervised"] for e in examples), \
        "the shift lost supervised positions; the first response token may not be trained"
    print(f"  loss          {reported.loss.item():.4f} == hand-computed response-only "
          f"{by_hand:.4f} (whole sequence would be {whole:.4f})")


def check_end_to_end(tok, tmp):
    """Run main() on a tiny model for a few steps, to catch anything the unit checks cannot."""
    base = os.path.join(tmp, "base")
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                               eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id,
                               pad_token_id=tok.pad_token_id)).save_pretrained(base)
    tok.save_pretrained(base)

    data = os.path.join(tmp, "pairs.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        for i in range(400):
            pair = dict(PAIRS[i % len(PAIRS)])
            pair["enstriksyon"] = f"{pair['enstriksyon']} ({i})"
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    out = os.path.join(tmp, "run")
    sys.argv = ["train_sft.py", "--data", data, "--base", base, "--out", out, "--epochs", "1",
                "--batch-size", "4", "--grad-accum", "2", "--eval-every", "10", "--sample-every", "0",
                "--val-share", "0.1", "--device", "cpu", "--warmup", "2"]
    train_sft.main()

    with open(os.path.join(out, "sft_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["steps"] > 0, summary
    assert summary["after"]["eval_loss"] < summary["before"]["eval_loss"], \
        f"loss did not fall: {summary['before']} -> {summary['after']}"
    assert os.path.exists(os.path.join(out, "model.safetensors"))
    print(f"  end to end    {summary['steps']} steps, held-out loss "
          f"{summary['before']['eval_loss']:.3f} -> {summary['after']['eval_loss']:.3f}, model saved")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="corpus/tokenizer")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer {args.tokenizer}: {len(tok):,} tokens, eos {tok.eos_token_id}, pad {tok.pad_token_id}")
    check_masking(tok)
    check_refuses_to_truncate(tok)
    check_collate(tok)
    check_batches()
    check_split()
    check_loss_ignores_prompt(tok)
    with tempfile.TemporaryDirectory() as tmp:
        check_end_to_end(tok, tmp)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
