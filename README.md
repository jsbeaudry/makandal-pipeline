# Makandal pipeline

Building a Haitian Creole language model from scratch: corpus, tokenizer, pretraining, evaluation, and
instruction tuning. Every number in this README was measured by the scripts in it.

## What happened to the first model

[jsbeaudry/makandal-pre-trained](https://huggingface.co/jsbeaudry/makandal-pre-trained) was trained with
`max_length=64, truncation=True` on a dataset loaded with `load_dataset("text", ...)`, which yields **one
row per line**. The corpus is 490 lines, so **31,360 of its 901,369 tokens ever reached the model — 3.5%**.
Read straight from the published weights: position rows 64–1023 still hold their initial values, because a
position that never appears in a training sequence never receives a gradient.

Two further faults: `config.json` declares `eos_token_id` 50256, which does not exist in a 33,977-token
vocabulary, so the model can never stop; and `eval_steps` was set without an eval strategy, so validation
never ran and a training loss of 0.0193 looked like success.

## The rebuild

| | Published | Rebuilt |
|---|---|---|
| Bits per character, held out | 4.526 | **1.129** |
| Position rows trained | 64 of 1,024 | **1,024 of 1,024** |
| Training tokens | 31,360 | **466,421,760** |
| Tokenizer fertility | 1.406 tok/word | **1.296** |
| Ends a document by itself | 0 of 10 | 3 of 10 |

Bits per character is the honest comparison: the two models have different tokenizers, so perplexity per
token is not comparable between them.

## Corpus

`corpus/` surveys, builds, tokenises and packs the training data.

```bash
python3 corpus/survey_sources.py --json corpus/survey.json
python3 corpus/build_corpus.py --out corpus/data
python3 corpus/train_tokenizer.py --corpus corpus/data --out corpus/tokenizer --vocab 32768
python3 corpus/pack_corpus.py --corpus corpus/data --tokenizer corpus/tokenizer --out corpus/packed
```

Sources: FineWeb-2, HPLT 2.0, MADLAD-400, finepdfs, Wikipedia, GlotCC, Glot500. Deliberately excluded:
xP3x `hat_Latn` (it is FLORES and XCOPA, so training on it invalidates later benchmarks) and the Aya
Collection Haitian split (machine translation).

**Deduplication is the whole job.** Four of those sources are CommonCrawl, so the same page arrives three
times. Splitting on blank lines does almost nothing — 0% of HPLT documents and 4% of FineWeb-2 documents
contain one, and MADLAD-400 arrives as a single line per document — so `build_corpus.py` deduplicates on
lines where documents have lines and on sentences where they do not. That took the corpus from 2,249M
characters to 1,833M, and revealed that **94% of Glot500 documents repeat text another source already
contributed**.

## Pretraining

```bash
python3 pipeline/train.py --data corpus/packed --out runs/base --epochs 4 \
    --batch-size 8 --grad-accum 6 --lr 6e-4 --checkpoint-every 1000 --max-hours 17
python3 pipeline/compare.py --models jsbeaudry/makandal-pre-trained jsbeaudry/makandal-base \
    --documents corpus/data
```

A plain PyTorch loop, no Trainer: AdamW with decay only on matrices, cosine schedule with warmup,
gradient clipping, a held-out check every `--eval-every` steps, the best weights saved rather than the
last, checkpoint and resume, and a wall-clock budget. It aborts on the first non-finite loss instead of
burning the rest of the schedule.

`pipeline/evaluate.py` reports what a training loss cannot: loss by position with each document fed from
its own first token, the same 64-token window taken from a document start and from mid-document (which
separates "context used badly" from "only beginnings trained"), how much of a training document the model
recites, and how many position rows still hold their initial values.

## Instruction tuning

```bash
python3 corpus/generate_sft.py --target 30000 --workers 24 --out sft.jsonl
```

A teacher model writes instruction and response pairs across a grid of 40 domains x 8 task types x 12
angles. A 200-pair pilot settled the prompt language: prompting in Kreyòl and in English give the same
function-word rate (0.396 vs 0.403), but the Kreyòl prompt kept accents on 100 of 100 responses against
97 of 100, and the student's tokenizer learned accented forms. Pairs are filtered as they are generated —
length, Kreyòl function-word floor, French ceiling, teacher commentary, duplicates — so the log shows what
is being thrown away while there is still time to react.

## Measuring Kreyòl

```bash
python3 eval/kreyol_bench.py Qwen/Qwen3-1.7B runs/kreyol-sft --limit 150
```

Three scores, because they fail independently: **XCOPA ht** accuracy for whether a model understands
Kreyòl (two-way, so 50 is chance), **FLORES+ chrF++** for whether it can produce Kreyòl (character
level, because a correct sentence should not be punished for choosing another spelling), and the
function-word rate on free generation for whether it answers in Kreyòl at all rather than drifting
into French.

**These benchmarks are clean here and almost nowhere else.** xP3x `hat_Latn` *is* FLORES and XCOPA, and
it sits in most multilingual instruction mixes, so a model trained on it has already seen the test set.
`build_corpus.py` excludes xP3x for exactly this reason. That decision is what makes these numbers
worth reporting, and it means a score from a model whose training data you do not know is not
comparable to one of these.

FLORES+ is used rather than `facebook/flores`, which is gated: the same 1,012 devtest sentences,
aligned across languages by id, which the script asserts before scoring.

## Running on a pod

`pod/` holds entry points that clone this repository, so no code has to travel through an API field:

```bash
bash -c "curl -fsSL https://raw.githubusercontent.com/jsbeaudry/makandal-pipeline/main/pod/run_sft_generation.sh | bash"
```

Two things bite on rented GPUs. Installing vLLM pulls a torch built for CUDA 13, so an older host fails at
engine init with `CUDA unknown error` — ask for a host with a CUDA 13 driver. And community hosts
occasionally hand back a GPU torch cannot initialise at all, so jobs that do not need one should fall
back to CPU rather than dying.
