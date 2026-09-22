"""Turn the surveyed sources into one deduplicated Kreyòl corpus, and account for every document dropped.

The sources overlap heavily: FineWeb-2, HPLT, MADLAD-400 and GlotCC are all CommonCrawl, so the same page
often appears in three of them. Deduplication is therefore the whole job, and it happens twice: once per
document, once per paragraph, both across sources. Paragraph deduplication is what removes boilerplate —
the cookie banner repeated on ten thousand pages.

Low-resource language labels are also noisy, so every document is checked against Kreyòl function words and
dropped if it looks more like French, which is the usual contaminant.

    python3 corpus/build_corpus.py --out corpus/data --sources wikipedia            # a small test
    python3 corpus/build_corpus.py --out corpus/data --limit-per-source 5000        # a wider test
    python3 corpus/build_corpus.py --out corpus/data                                # everything

Output is `shard-XXXX.jsonl.gz`, one JSON document per line, plus `manifest.json` with per-source counts
and the reason every dropped document was dropped.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

# Kreyòl function words, and the French ones that give away a misfiled French page.
KREYOL = set("yo nan pou ak se li ki gen pa mwen nou te ap sa yon lan la men epi sou tout kòm konsa fè di "
             "kap ale vin bay anpil lè yon ou pral kote poukisa paske avèk nap".split())
FRENCH = set("les des est une dans pour avec sont cette nous vous leur mais plus tout aussi être avoir "
             "que qui par sur ses ces elle ils".split())
WORD = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ']+")
# Pages that say outright they were machine translated. This catches only the ones that admit it; crawls
# for a low-resource language carry more undeclared machine translation, which no cheap filter finds.
TRANSLATED = re.compile(r"tradui\w* otomatikman|tradiksyon otomatik|google transl\w+|translate\.google"
                        r"|powered by translate|automatically translated|machine[- ]translat\w+"
                        r"|traduit automatiquement|traducci[oó]n autom[aá]tica")

SOURCES = {
    # name:        (dataset, config, split, sentence level?)
    "fineweb2":    ("HuggingFaceFW/fineweb-2", "hat_Latn", "train", False),
    "hplt":        ("HPLT/HPLT2.0_cleaned", "hat_Latn", "train", False),
    "finepdfs":    ("HuggingFaceFW/finepdfs", "hat_Latn", "train", False),
    "wikipedia":   ("wikimedia/wikipedia", "20231101.ht", "train", False),
    "glotcc":      ("cis-lmu/GlotCC-V1", "hat-Latn", "train", False),
    "glot500":     ("cis-lmu/Glot500", "hat_Latn", "train", True),
    "madlad":      ("allenai/MADLAD-400", "data/ht/ht_clean_0000.jsonl.gz", "train", False),
}
ORDER = ["fineweb2", "hplt", "madlad", "finepdfs", "wikipedia", "glotcc", "glot500"]


class Bloom:
    """A set that answers 'have I seen this?' in 8 bits per item instead of 60 bytes."""

    def __init__(self, capacity, error=0.01):
        self.m = max(1024, int(-capacity * math.log(error) / (math.log(2) ** 2)))
        self.k = max(1, round(self.m / capacity * math.log(2)))
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.added = 0

    def add_if_new(self, data):
        h = int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "little")
        a, b = h & 0xFFFFFFFFFFFFFFFF, (h >> 64) | 1
        spots = [(a + i * b) % self.m for i in range(self.k)]
        if all((self.bits[s >> 3] >> (s & 7)) & 1 for s in spots):
            return False
        for s in spots:
            self.bits[s >> 3] |= 1 << (s & 7)
        self.added += 1
        return True

    @property
    def megabytes(self):
        return self.bits.nbytes / 1e6


def looks_kreyol(words, floor=0.06):
    if len(words) < 20:
        return False
    kreyol = sum(w in KREYOL for w in words) / len(words)
    french = sum(w in FRENCH for w in words) / len(words)
    return kreyol >= floor and kreyol > french


def quality(text, words):
    """Gopher-style checks, in the order that rejects most cheaply."""
    if len(text) < 200:
        return "too short"
    if len(words) < 30:
        return "too few words"
    mean_word = sum(len(w) for w in words) / len(words)
    if not 2.0 <= mean_word <= 12.0:
        return "word length"
    letters = sum(c.isalpha() for c in text) / len(text)
    if letters < 0.6:
        return "not enough letters"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.75:
        return "repeated lines"
    if (text.count("#") + text.count("…")) / max(len(words), 1) > 0.1:
        return "symbols"
    return None


def segments(text):
    """The unit deduplication works on.

    Splitting on blank lines does almost nothing here: 0% of HPLT documents and 4% of FineWeb-2 documents
    contain one, and MADLAD-400 arrives as a single line per document. So split on lines where there are
    lines, and on sentences where there is only one, which is what makes the same page arriving from two
    crawls actually collide.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines, "\n"
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()], " "


def clean(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # A navigation menu arrives as one long line with its spaces stripped out; prose runs about one
    # space every six characters. Dropping the line keeps the rest of the page, which is usually fine.
    lines = [l for l in text.split("\n")
             if not (len(l.strip()) > 120 and l.count(" ") / max(len(l.strip()), 1) < 0.05)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_source(name, limit, cache_dir):
    """Yield raw text from one source, streaming so nothing large lands on disk."""
    dataset, config, split, sentences = SOURCES[name]
    if name == "madlad":                                      # not served as parquet; fetch the one file
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(dataset, config, repo_type="dataset", cache_dir=cache_dir)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    return
                try:
                    yield json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
        return

    from datasets import load_dataset
    stream = load_dataset(dataset, config, split=split, streaming=True)
    buffer = []
    for i, row in enumerate(stream):
        if limit and i >= limit:
            break
        text = next((row[f] for f in ("text", "content", "raw_content", "document") if row.get(f)), "")
        if not text:
            continue
        if sentences:                                         # glue sentences into workable documents
            buffer.append(text.strip())
            if sum(len(s) for s in buffer) >= 2000:
                yield "\n".join(buffer)
                buffer = []
        else:
            yield text
    if buffer:
        yield "\n".join(buffer)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="corpus/data")
    p.add_argument("--sources", default=",".join(ORDER), help="comma separated, in priority order")
    p.add_argument("--limit-per-source", type=int, default=0, help="rows to read, 0 for all")
    p.add_argument("--shard-docs", type=int, default=50_000)
    p.add_argument("--expect-docs", type=int, default=6_000_000, help="sizes the duplicate filter")
    p.add_argument("--expect-paragraphs", type=int, default=80_000_000)
    p.add_argument("--dedup-floor", type=int, default=60,
                   help="only deduplicate pieces at least this long, so common short lines survive")
    p.add_argument("--cache", default=None, help="where downloads land")
    args = p.parse_args()

    names = [n.strip() for n in args.sources.split(",") if n.strip() in SOURCES]
    os.makedirs(args.out, exist_ok=True)
    seen_docs = Bloom(args.expect_docs)
    seen_paragraphs = Bloom(args.expect_paragraphs)
    print(f"duplicate filters: {seen_docs.megabytes:.0f} MB for documents, "
          f"{seen_paragraphs.megabytes:.0f} MB for paragraphs\n", flush=True)

    stats, shard, in_shard, out = {}, 0, 0, None
    kept_chars = kept_docs = 0
    started = time.time()
    for name in names:
        counts = {"read": 0, "kept": 0, "chars": 0, "duplicate": 0, "not kreyol": 0}
        source_started = time.time()
        for text in read_source(name, args.limit_per_source, args.cache):
            counts["read"] += 1
            text = clean(text)
            words = WORD.findall(text.lower())
            why = quality(text, words)                        # length first, so the counts stay honest
            if why:
                counts[why] = counts.get(why, 0) + 1
                continue
            if not looks_kreyol(words):
                counts["not kreyol"] += 1
                continue
            if TRANSLATED.search(text.lower()):
                counts["machine translated"] = counts.get("machine translated", 0) + 1
                continue
            if not seen_docs.add_if_new(hashlib.blake2b(text.encode(), digest_size=16).digest()):
                counts["duplicate"] += 1
                continue
            pieces, separator = segments(text)
            # Only long pieces are deduplicated: a short line like "Amen." is legitimately everywhere,
            # and removing every repeat of it would gut the documents that follow.
            fresh = [q for q in pieces
                     if len(q) < args.dedup_floor or seen_paragraphs.add_if_new(q.encode())]
            if not fresh or sum(len(q) for q in fresh) < 200:
                counts["duplicate"] += 1
                continue
            if len(fresh) < len(pieces):
                counts["partly duplicate"] = counts.get("partly duplicate", 0) + 1
            text = separator.join(fresh)
            if out is None or in_shard >= args.shard_docs:
                if out:
                    out.close()
                out = gzip.open(os.path.join(args.out, f"shard-{shard:04d}.jsonl.gz"), "wt", encoding="utf-8")
                shard, in_shard = shard + 1, 0
            out.write(json.dumps({"text": text, "source": name}, ensure_ascii=False) + "\n")
            in_shard += 1
            counts["kept"] += 1
            counts["chars"] += len(text)
            kept_docs += 1
            kept_chars += len(text)
            if counts["read"] % 20_000 == 0:
                rate = counts["read"] / max(time.time() - source_started, 1)
                print(f"  {name}: read {counts['read']:,}  kept {counts['kept']:,}  "
                      f"{counts['chars']/1e6:,.0f}M chars  {rate:,.0f} rows/s", flush=True)
        stats[name] = counts
        print(f"{name}: kept {counts['kept']:,} of {counts['read']:,} documents, "
              f"{counts['chars']/1e6:,.0f}M characters "
              f"(~{counts['chars']/3.6/1e6:,.0f}M tokens) in {(time.time()-source_started)/60:.1f} min",
              flush=True)
        print("   dropped: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()
                                         if k not in ("read", "kept", "chars") and v), flush=True)
    if out:
        out.close()

    manifest = {"documents": kept_docs, "characters": kept_chars,
                "estimated_tokens": int(kept_chars / 3.6), "shards": shard,
                "minutes": round((time.time() - started) / 60, 1), "sources": stats}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{kept_docs:,} documents, {kept_chars/1e6:,.0f}M characters "
          f"(~{kept_chars/3.6/1e6:,.0f}M tokens) in {shard} shards -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
