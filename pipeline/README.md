# Makandal pipeline

Three scripts that prepare data, train, and score a model on text it has never seen. They replace the
notebook that produced [jsbeaudry/makandal-pre-trained](https://huggingface.co/jsbeaudry/makandal-pre-trained),
whose training used 3% of the corpus and a context window of 64 tokens.

```bash
python3 pipeline/prepare_data.py --input "data/*.txt" --out pipeline/prepared --block 1024
python3 pipeline/train.py --data pipeline/prepared --out pipeline/runs/proof --epochs 6 --dropout 0.1
python3 pipeline/evaluate.py --model pipeline/runs/proof --data pipeline/prepared
```

## What was wrong

Every number below is measured on the published weights, not inferred from the notebook.

**Only the first 64 tokens of each document were ever trained.** Fed a document from its first token, the
model predicts well up to token 64 and collapses to chance immediately after. Chance for a 33,977-token
vocabulary is a loss of 10.43.

| Position in the document | Loss | Next token guessed right |
|---|---|---|
| tokens 0–32 | 4.42 | 41.8% |
| tokens 32–64 | 3.26 | 41.8% |
| tokens 64–128 | 8.65 | 12.5% |
| tokens 128–256 | 9.36 | 7.6% |
| tokens 256–512 | 9.82 | 6.4% |
| tokens 512–1024 | 9.61 | 6.9% |

Two independent checks confirm the cause is truncation at 64 tokens rather than a context the model simply
uses badly:

- a standalone 64-token window taken from the **middle** of a training document (token 600) scores 9.54,
  against 4.25 for a 64-token window from the **start** of the same documents. Only document beginnings
  were ever seen;
- in the weights themselves, position rows 64–1023 still sit at their initial spread (0.0199, flat across
  all of them), while rows 0–63 have moved. A position that never appears in a training sequence never
  receives a gradient.

That makes the trained corpus 414 × 64 = **26,496 tokens of the 824,320 the corpus holds**. With 5,600
steps at batch 8, the whole run saw 2.9M tokens of signal; a 112M-parameter model wants roughly 2.2B.

**The published weights are not the end-of-training weights.** The card reports a final training loss of
0.0193 at step 5,600. Measured on exactly the tokens it trained on, the published checkpoint scores 3.98 —
which is the card's own step-200 row (4.2640 at step 200, 3.8550 at step 250). Two related claims in the
old version of this file were wrong and are withdrawn: the model does not recite its training data. Given
the first 32 tokens of a training document it copies **0.3 words** before diverging, and prompting it with
`"Kontabilite se"` returns fluent but invented text, not the corpus opening.

**It can never stop.** `config.json` declares `eos_token_id` and `bos_token_id` 50256, an id outside a
33,977-token vocabulary, so the token cannot be emitted; the training text had no document separators
either, so nothing taught the model to end a document.

## What these scripts do instead

**`prepare_data.py`** reads whole documents, drops identical and near-identical ones (13-word shingles),
holds out a share **by document**, puts `<|endoftext|>` between documents, concatenates, and cuts the stream
into blocks of `--block` tokens. Nothing is truncated. On this corpus: 407 documents after dedup,
**824,320 tokens packed** against the old 26,496.

**`train.py`** is a plain PyTorch loop — no Trainer, no accelerate — so each part is visible: AdamW with
decay only on matrices, cosine schedule with warmup, gradient accumulation, gradient clipping, a held-out
check every `--eval-every` steps, early stopping on that check, and the best weights saved rather than the
last. It sets eos/bos/pad to the ids this tokenizer really uses. It runs on a Mac (MPS, fp32) and on a GPU
(CUDA, bfloat16) unchanged. `--init <model>` continues from existing weights.

**`evaluate.py`** answers what a training loss cannot:

- perplexity on held-out documents, on training documents, and on any other Kreyòl file (`--other`), which
  is how you see whether the model learned the language or just this corpus;
- loss and next-token accuracy by position, each document fed from its own first token — flat means the
  context is used, a cliff means it is not;
- the same 64-token window taken from a document start and from mid-document, which separates "context
  used badly" from "only beginnings trained";
- memorisation: how many words of a training document it recites when given the first 32 tokens;
- how many position rows are still at their initial values, read straight from the weights;
- samples, and whether the model stops on its own.

## Reading the numbers

This corpus is 0.9M tokens and the model is 112M parameters, which wants roughly 2.2B. No pipeline fixes
that. A proof run here shows the mechanics work — the whole block trains, held-out loss is measurable, the
model stops at a document end — and held-out perplexity will still be poor. The corpus is also GPT-4 output
on 414 school subjects, so the model learns that register and those mistakes; real Kreyòl (web text, books,
transcripts) has to carry the weight.

Both scripts print their assumptions, and `training_summary.json` in the run folder keeps the curve, the
token counts and the best held-out loss, so a model card can quote measured numbers.
