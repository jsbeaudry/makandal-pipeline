"""Score a model on Kreyòl against benchmarks this project has kept clean.

Three questions, because they fail independently and a single number hides which one broke:

  Does it understand Kreyòl?   XCOPA ht, accuracy. Two choices, the one the model finds more likely.
  Can it produce Kreyòl?       FLORES-200, English to Kreyòl, chrF++. Character-level, because word
                               overlap punishes a correct sentence for choosing another spelling.
  Does it answer in Kreyòl?    Free generation, scored by function-word rate and French rate — the
                               same measure corpus/generate_sft.py filters with.

**These benchmarks are clean for this project and for almost nobody else.** xP3x `hat_Latn` is FLORES
and XCOPA, and it is in most multilingual instruction mixes; a model trained on it has seen the test
set. corpus/build_corpus.py excludes xP3x for exactly this reason, which is what makes these numbers
mean something here. A score from a model whose training data you do not know is not comparable to one
of these, and saying so is not pedantry — it is the difference between a measurement and a decoration.

    python3 eval/kreyol_bench.py Qwen/Qwen3-1.7B
    python3 eval/kreyol_bench.py Qwen/Qwen3-1.7B runs/kreyol-sft --limit 200 --device mps
"""
import argparse
import json
import re
import sys
import time
import unicodedata

import torch

# FLORES+ rather than facebook/flores, which is gated: the successor, same 1,012 devtest sentences,
# aligned across languages by id. XCOPA publishes 100 validation items for ht; test is unlabelled.
FLORES = "openlanguagedata/flores_plus"
XCOPA = "cambridgeltl/xcopa"

KREYOL_WORDS = set(("yo nan pou ak se li ki gen pa mwen nou te ap sa yon lan la men epi sou tout kom "
                    "konsa fe di kap ale vin bay anpil le ou pral kote poukisa paske avek nap chak").split())
FRENCH_WORDS = set(("les des est une dans pour avec sont cette nous vous leur mais plus tout aussi etre "
                    "avoir que qui par sur ses ces elle ils vos votre nos").split())
WORD = re.compile(r"[a-z']+")

# Free-generation prompts, written here rather than drawn from any dataset, so no model has seen them.
PROMPTS = [
    "Ki sa yon moun ta dwe fè si yon siklòn ap vini?",
    "Eksplike yon timoun sa ki fè lapli tonbe.",
    "Bay twa etap pou plante yon pye mango.",
    "Reekri fraz sa a pou l pi poli: 'Ban m kòb mwen an kounye a!'",
    "Yon kliyan mande konbyen pri diri a. Reponn li tankou yon machann nan mache a.",
    "Ki sa ki enpòtan pou yon elèv fè anvan yon egzamen?",
]


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def rates(text):
    """How Kreyòl this reads, and how French, as shares of its words."""
    words = WORD.findall(fold(text))
    if not words:
        return 0.0, 1.0
    return (sum(w in KREYOL_WORDS for w in words) / len(words),
            sum(w in FRENCH_WORDS for w in words) / len(words))


def load(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16 if device != "cpu" else torch.float32).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def ask(tok, model, prompt, device, new_tokens=140):
    """One turn. Qwen3 reasons before answering unless told not to, which is not what is measured."""
    if getattr(tok, "chat_template", None):
        try:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
    else:
        text = prompt
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def logprob(tok, model, context, continuation, device):
    """Mean log-probability of `continuation` given `context`, for picking between two choices."""
    whole = tok(context + continuation, return_tensors="pt").to(device)
    prefix = len(tok(context)["input_ids"])
    logits = model(**whole).logits[0, :-1]
    targets = whole["input_ids"][0, 1:]
    keep = slice(max(prefix - 1, 0), None)
    chosen = torch.log_softmax(logits[keep].float(), dim=-1).gather(
        1, targets[keep].unsqueeze(1)).squeeze(1)
    return float(chosen.mean()) if chosen.numel() else float("-inf")


def run_xcopa(tok, model, device, limit):
    """Two candidate causes or effects; the model is right if it prefers the labelled one."""
    from datasets import load_dataset
    rows = load_dataset(XCOPA, "ht", split="validation")
    rows = rows.select(range(min(limit, len(rows))))
    right = 0
    for row in rows:
        joiner = "paske" if row["question"] == "cause" else "donk"
        context = f"{row['premise'].rstrip('.')} {joiner} "
        scores = [logprob(tok, model, context, c[0].lower() + c[1:], device)
                  for c in (row["choice1"], row["choice2"])]
        right += int((0 if scores[0] > scores[1] else 1) == row["label"])
    return {"n": len(rows), "accuracy": round(100 * right / max(len(rows), 1), 1)}


def run_flores(tok, model, device, limit):
    """English into Kreyòl, scored with chrF++ because spelling varies and words do not overlap."""
    import sacrebleu
    from datasets import load_dataset
    src = load_dataset(FLORES, "eng_Latn", split="devtest")
    ref = load_dataset(FLORES, "hat_Latn", split="devtest")
    assert src[0]["id"] == ref[0]["id"], "the two languages are not aligned by id"
    n = min(limit, len(src))
    hyps, refs = [], []
    for i in range(n):
        english = src[i]["text"]
        out = ask(tok, model, f"Tradui fraz sa a an kreyòl ayisyen. Bay tradiksyon an sèlman.\n\n"
                              f"{english}", device, new_tokens=120)
        hyps.append(" ".join(out.split("\n")[0].split()))
        refs.append(ref[i]["text"])
    return {"n": n,
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
            "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "sample": {"english": src[0]["text"][:90], "got": hyps[0][:90], "want": refs[0][:90]}}


def run_free(tok, model, device):
    """Asked in Kreyòl, does it answer in Kreyòl — or drift into French or English?"""
    kreyol, french, answers = 0.0, 0.0, []
    for prompt in PROMPTS:
        out = ask(tok, model, prompt, device)
        k, f = rates(out)
        kreyol += k
        french += f
        answers.append({"q": prompt, "a": " ".join(out.split())[:200], "kreyol": round(k, 3)})
    return {"n": len(PROMPTS), "kreyol_word_rate": round(kreyol / len(PROMPTS), 3),
            "french_word_rate": round(french / len(PROMPTS), 3), "answers": answers}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="model ids or folders; the first is the baseline")
    p.add_argument("--limit", type=int, default=150, help="items per benchmark")
    p.add_argument("--device", default=None, help="cuda, mps or cpu")
    p.add_argument("--skip", nargs="*", default=[], choices=["xcopa", "flores", "free"])
    p.add_argument("--out", default="eval/kreyol_bench.json")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             ("mps" if torch.backends.mps.is_available() else "cpu"))
    report = {}
    for name in args.models:
        print(f"\n=== {name} on {device} ===", flush=True)
        started = time.time()
        tok, model = load(name, device)
        row = {}
        for kind, fn in (("xcopa", lambda: run_xcopa(tok, model, device, args.limit)),
                         ("flores", lambda: run_flores(tok, model, device, args.limit)),
                         ("free", lambda: run_free(tok, model, device))):
            if kind in args.skip:
                continue
            try:
                row[kind] = fn()
                print(f"  {kind}: {json.dumps({k: v for k, v in row[kind].items() if k not in ('answers', 'sample')})}",
                      flush=True)
            except Exception as error:
                row[kind] = {"error": f"{type(error).__name__}: {error}"}
                print(f"  {kind} failed: {row[kind]['error']}", flush=True)
        row["minutes"] = round((time.time() - started) / 60, 1)
        report[name] = row
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print("\n| model | XCOPA ht | FLORES chrF++ | BLEU | Kreyòl words | French words |")
    print("|---|---|---|---|---|---|")
    for name, row in report.items():
        def cell(kind, key):
            got = row.get(kind, {})
            return "—" if "error" in got or key not in got else got[key]
        print(f"| {name} | {cell('xcopa', 'accuracy')} | {cell('flores', 'chrf')} | "
              f"{cell('flores', 'bleu')} | {cell('free', 'kreyol_word_rate')} | "
              f"{cell('free', 'french_word_rate')} |")
    print(f"\nwritten to {args.out}")
    print("XCOPA is two-way: 50 is chance, not 0.")


if __name__ == "__main__":
    main()
