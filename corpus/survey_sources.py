"""Ask the Hub how much Haitian Creole each candidate source actually holds, before downloading any of it.

Every figure comes from the datasets-server, not from a dataset card. Bytes are the stored parquet size;
tokens are estimated from bytes with a measured bytes-per-token ratio for Kreyòl, which is close enough to
decide what is worth downloading and what is not.

    python3 corpus/survey_sources.py
    python3 corpus/survey_sources.py --json corpus/survey.json

Two sources are deliberately absent. xP3x `hat_Latn` is FLORES and XCOPA, which are evaluation sets, and
training on them would make every later benchmark meaningless. The Aya Collection's Haitian split is machine
translation output, which teaches a model translationese rather than Kreyòl.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

SOURCES = [
    # (dataset, config, what it is)
    ("HuggingFaceFW/fineweb-2", "hat_Latn", "web crawl, filtered and deduplicated"),
    ("HuggingFaceFW/finepdfs", "hat_Latn", "text extracted from PDFs"),
    ("HPLT/HPLT2.0_cleaned", "hat_Latn", "web crawl, cleaned"),
    ("allenai/madlad-400", "ht", "web crawl, audited by language"),
    ("wikimedia/wikipedia", "20231101.ht", "encyclopedia"),
    ("statmt/cc100", "ht", "CommonCrawl, one language per file"),
    ("oscar-corpus/OSCAR-2301", "ht", "CommonCrawl, classified"),
    ("cis-lmu/Glot500", "hat_Latn", "mixed sources for low-resource languages"),
    ("cis-lmu/GlotCC-V1", "hat-Latn", "CommonCrawl for low-resource languages"),
    # MADLAD-400 and CC-100 are not served by the datasets-server, so build_corpus.py fetches their
    # files directly; MADLAD-400's ht clean split is 176 MB gzipped, roughly 165M tokens.
    ("facebook/flores", "hat_Latn", "EVALUATION ONLY - never train on this"),
]

SERVER = "https://datasets-server.huggingface.co"


def ask(path, params):
    url = f"{SERVER}/{path}?" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.load(e).get("error", "")[:60]
        except Exception:
            pass
        return None, f"HTTP {e.code} {detail}".strip()
    except Exception as e:                                   # network, timeout, malformed JSON
        return None, str(e)[:60]


def configs_for(dataset):
    data, err = ask("splits", {"dataset": dataset})
    if not data:
        return [], err
    return sorted({s["config"] for s in data.get("splits", [])}), None


def size_of(dataset, config):
    data, err = ask("size", {"dataset": dataset, "config": config})
    if not data:
        return None, err
    size = data.get("size", {}).get("config", {})
    # num_bytes_memory is the uncompressed text; the parquet figure understates a corpus by about 2.5x
    return {"rows": size.get("num_rows"),
            "bytes": size.get("num_bytes_memory") or size.get("num_bytes_parquet_files")}, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bytes-per-token", type=float, default=3.6,
                   help="Kreyol averages about 3.6 bytes of UTF-8 per token with a Kreyol BPE")
    p.add_argument("--json", help="also write the findings here")
    args = p.parse_args()

    print(f"{'source':38s} {'config':14s} {'rows':>10s} {'size':>10s} {'~tokens':>12s}  note")
    print("-" * 110)
    found, total_tokens = [], 0
    for dataset, config, note in SOURCES:
        size, err = size_of(dataset, config)
        if err or not size or not size.get("bytes"):
            names, list_err = configs_for(dataset)
            hint = ""
            if names:
                near = [n for n in names if any(k in n.lower() for k in ("ht", "hat", "cre"))]
                hint = f"configs: {', '.join(near[:4])}" if near else f"{len(names)} configs, none Kreyol"
            print(f"{dataset[:38]:38s} {config[:14]:14s} {'-':>10s} {'-':>10s} {'-':>12s}  "
                  f"{err or list_err or 'no size'} {hint}")
            continue
        tokens = int(size["bytes"] / args.bytes_per_token)
        evaluation = "EVALUATION" in note
        if not evaluation:
            total_tokens += tokens
        found.append({"dataset": dataset, "config": config, "rows": size["rows"],
                      "bytes": size["bytes"], "tokens": tokens, "note": note, "evaluation": evaluation})
        print(f"{dataset[:38]:38s} {config[:14]:14s} {size['rows'] or 0:>10,} "
              f"{size['bytes']/1e6:>9,.0f}M {tokens:>12,}  {note}")

    print("-" * 110)
    print(f"{'usable total':38s} {'':14s} {'':>10s} {'':>10s} {total_tokens:>12,}")
    print(f"\na 112M-parameter model wants about 2.2B tokens; {total_tokens/2.24e9:.0%} of that is here.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"bytes_per_token": args.bytes_per_token, "usable_tokens": total_tokens,
                       "sources": found}, f, indent=1)
        print(f"written to {args.json}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
