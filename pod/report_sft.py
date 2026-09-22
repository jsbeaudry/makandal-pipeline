"""Summarise a generated instruction set: how Kreyol it is, how varied, and what it looks like."""
import collections
import json
import re
import sys
import unicodedata

KREYOL = set(('yo nan pou ak se li ki gen pa mwen nou te ap sa yon lan la men epi sou tout kom konsa '
              'fe di kap ale vin bay anpil le ou pral kote poukisa paske avek nap chak').split())
FRENCH = set(('les des est une dans pour avec sont cette nous vous leur mais plus tout aussi etre '
              'avoir que qui par sur ses ces elle ils vos votre nos').split())
WORD = re.compile(r"[a-z']+")


def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


def main(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    kreyol = french = 0.0
    for row in rows:
        words = WORD.findall(fold(row['enstriksyon'] + ' ' + row['repons']))
        if words:
            kreyol += sum(w in KREYOL for w in words) / len(words)
            french += sum(w in FRENCH for w in words) / len(words)
    count = max(len(rows), 1)
    print('[pod] REPORT ' + json.dumps({
        'pairs': len(rows),
        'kreyol_word_rate': round(kreyol / count, 4),
        'french_word_rate': round(french / count, 4),
        'with_accents': sum(1 for r in rows if any(ord(c) > 127 for c in r['repons'])),
        'unique_instructions': len(set(r['enstriksyon'] for r in rows)),
        'domains': len(set(r['domain'] for r in rows)),
        'tasks': len(set(r['task'] for r in rows)),
        'median_response_chars': sorted(len(r['repons']) for r in rows)[len(rows) // 2],
    }), flush=True)
    per_task = collections.Counter(r['task'][:32] for r in rows)
    for task, n in per_task.most_common():
        print(f'[pod] task {n:6,}  {task}', flush=True)
    for row in rows[:5]:
        print('[pod] SAMPLE | ' + row['domain'] + ' | ' + row['enstriksyon'][:170], flush=True)
        print('[pod]     -> ' + row['repons'][:280], flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/workspace/sft.jsonl')
