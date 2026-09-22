"""Train a byte-level BPE on the real corpus, and measure whether it beats the published one.

The published tokenizer was trained on 4.7 MB of one writer's output, so it is efficient on that register
and wasteful everywhere else. This trains on the built corpus and reports fertility — tokens per word — for
both, on text neither tokenizer was trained on. Fewer tokens per word means more Kreyòl fits in the same
context and the same compute buys more text.

    python3 corpus/train_tokenizer.py --corpus corpus/data --out corpus/tokenizer --vocab 32768

Special tokens match what train.py expects: <|endoftext|> first, then <|pad|>, then <|unk|>.
"""
import argparse
import glob
import gzip
import json
import os
import re

WORD = re.compile(r"\S+")


def documents(folder, limit=0):
    """Stream text out of the shards written by build_corpus.py."""
    count = 0
    for path in sorted(glob.glob(os.path.join(folder, "shard-*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                count += 1
                if limit and count >= limit:
                    return


def fertility(tokenizer, texts):
    """Tokens per whitespace word. Lower is better."""
    tokens = sum(len(ids) for ids in tokenizer(texts)["input_ids"])
    words = sum(len(WORD.findall(text)) for text in texts)
    return tokens / max(words, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="corpus/data")
    p.add_argument("--out", default="corpus/tokenizer")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--train-docs", type=int, default=400_000, help="documents used to fit the merges")
    p.add_argument("--compare", default="jsbeaudry/makandal-pre-trained", help="the tokenizer to beat")
    p.add_argument("--holdout-docs", type=int, default=2_000, help="documents used only for fertility")
    args = p.parse_args()

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import GPT2TokenizerFast

    holdout = []
    for i, text in enumerate(documents(args.corpus)):
        if i >= args.holdout_docs:
            break
        holdout.append(text)
    print(f"{len(holdout):,} documents held out for the fertility comparison", flush=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab,
                                  special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def corpus_iterator():
        for i, text in enumerate(documents(args.corpus, args.train_docs + args.holdout_docs)):
            if i < args.holdout_docs:                        # never fit on the comparison set
                continue
            if i % 20_000 == 0 and i:
                print(f"  fitting on document {i:,}", flush=True)
            yield text

    print(f"training a {args.vocab:,}-token BPE", flush=True)
    tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))
    fast = GPT2TokenizerFast(tokenizer_file=os.path.join(args.out, "tokenizer.json"),
                             eos_token="<|endoftext|>", bos_token="<|endoftext|>",
                             pad_token="<|pad|>", unk_token="<|unk|>")
    fast.save_pretrained(args.out)
    print(f"saved to {args.out} ({len(fast):,} tokens)", flush=True)

    print("\nfertility, tokens per word, on documents neither tokenizer was fitted on (lower is better)")
    mine = fertility(fast, holdout)
    print(f"  this tokenizer ({len(fast):,} tokens)   {mine:.3f}")
    try:
        from transformers import AutoTokenizer
        old = AutoTokenizer.from_pretrained(args.compare)
        theirs = fertility(old, holdout)
        print(f"  {args.compare[:34]:34s} {theirs:.3f}")
        print(f"  the same text costs {100*(mine/theirs - 1):+.1f}% tokens")
    except Exception as e:
        print(f"  could not load {args.compare}: {str(e)[:80]}")
    with open(os.path.join(args.out, "fertility.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": len(fast), "fertility": mine, "holdout_documents": len(holdout)}, f, indent=1)


if __name__ == "__main__":
    main()
