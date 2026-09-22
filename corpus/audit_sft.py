"""Measure what the on-pod report cannot: near-duplicate density and lexical diversity.

The generator deduplicates on an exact folded 160-character prefix, so 'unique_instructions: 30000'
is true by construction and says nothing. Two instructions that mean the same thing in different
words both survive. MinHash over word trigrams finds those.
"""
import collections
import json
import re
import sys
import unicodedata

import numpy as np

PUNCT = re.compile(r"[^a-z0-9' ]+")


def fold(text):
    stripped = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                       if unicodedata.category(c) != 'Mn')
    return PUNCT.sub(' ', stripped).split()


def shingles(words, n=3):
    if len(words) < n:
        return {' '.join(words)} if words else set()
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def near_duplicate_rate(texts, threshold=0.7, hashes=64, bands=16):
    """Share of items having at least one other item with Jaccard >= threshold over word trigrams."""
    sets = [shingles(fold(t)) for t in texts]
    universe = {s: i for i, s in enumerate({s for row in sets for s in row})}
    rng = np.random.default_rng(0)
    a = rng.integers(1, 2 ** 31 - 1, hashes)
    b = rng.integers(0, 2 ** 31 - 1, hashes)
    prime = 2147483647

    signatures = np.full((len(sets), hashes), np.iinfo(np.int64).max, dtype=np.int64)
    for i, row in enumerate(sets):
        if not row:
            continue
        ids = np.array([universe[s] for s in row], dtype=np.int64)
        signatures[i] = ((a[None, :] * ids[:, None] + b[None, :]) % prime).min(axis=0)

    rows = hashes // bands
    candidates = set()
    for band in range(bands):
        buckets = collections.defaultdict(list)
        for i, sig in enumerate(signatures[:, band * rows:(band + 1) * rows]):
            buckets[sig.tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 60:          # a huge bucket is a hash collision, not 60 twins
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))

    flagged, pairs = set(), []
    for i, j in candidates:
        if not sets[i] or not sets[j]:
            continue
        overlap = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        if overlap >= threshold:
            flagged.add(i)
            flagged.add(j)
            pairs.append((overlap, i, j))
    return len(flagged) / max(len(texts), 1), sorted(pairs, reverse=True), len(candidates)


def distinct_n(texts, n):
    grams, total = set(), 0
    for text in texts:
        words = fold(text)
        for i in range(max(len(words) - n + 1, 0)):
            grams.add(' '.join(words[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


def main(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    instructions = [r['enstriksyon'] for r in rows]
    responses = [r['repons'] for r in rows]
    print(f'{len(rows):,} pairs')

    print('\n-- exact duplicates --')
    for name, texts in (('instruction', instructions), ('response', responses)):
        counts = collections.Counter(texts)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        print(f'  {name:12} {repeated:6,} exact repeats ({repeated / len(rows):.2%})')
    bags = collections.Counter(tuple(sorted(set(fold(t)))) for t in instructions)
    same_bag = sum(c - 1 for c in bags.values() if c > 1)
    print(f'  {"same wordbag":12} {same_bag:6,} instructions reuse another one\'s exact word set')

    print('\n-- near duplicates (word-trigram Jaccard >= 0.7) --')
    for name, texts in (('instruction', instructions), ('response', responses)):
        rate, pairs, checked = near_duplicate_rate(texts)
        print(f'  {name:12} {rate:.2%} of items, from {checked:,} candidate pairs')
        for overlap, i, j in pairs[:3]:
            print(f'      {overlap:.2f}  {texts[i][:88]}')
            print(f'            {texts[j][:88]}')

    print('\n-- lexical diversity --')
    for name, texts in (('instruction', instructions), ('response', responses)):
        print(f'  {name:12} distinct-1 {distinct_n(texts, 1):.4f}   '
              f'distinct-2 {distinct_n(texts, 2):.4f}   distinct-3 {distinct_n(texts, 3):.4f}')

    print('\n-- most repeated response openings (first 4 words) --')
    openings = collections.Counter(' '.join(fold(r)[:4]) for r in responses)
    for opening, n in openings.most_common(8):
        print(f'  {n:5,} ({n / len(rows):5.2%})  {opening}')

    lengths = sorted(len(r) for r in responses)
    print(f'\nresponse chars: p10 {lengths[len(lengths) // 10]}  median '
          f'{lengths[len(lengths) // 2]}  p90 {lengths[9 * len(lengths) // 10]}')


if __name__ == '__main__':
    main(sys.argv[1])
