"""Generate the Kreyol instruction set from the teacher model, in parallel, filtering as it goes.

The pilot settled two things: prompt in Kreyol (identical word rates, but 100/100 responses kept their
accents against 97/100), and ask for ten pairs per call, which the teacher returns as clean JSON.

Diversity comes from a grid of domain x task x angle, so 3,000 calls do not all ask the same thing.
Filtering happens during generation rather than after, so the log shows what is being thrown away while
there is still time to react.

    python3 corpus/generate_sft.py --target 30000 --workers 24 --out /workspace/sft.jsonl
"""
import argparse
import json
import random
import re
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DOMAINS = [
    'lavi chak jou', 'sante ak byennèt', 'agrikilti ak jaden', 'edikasyon ak lekòl',
    'biznis ak lajan', 'teknoloji ak telefòn', 'kilti ak istwa Ayiti', 'manje ak kizin',
    'dwa sitwayen', 'travay ak metye', 'fanmi ak timoun', 'vwayaj ak transpò',
    'mizik ak dans', 'espò ak jwèt', 'relijyon ak kwayans', 'anviwònman ak dlo',
    'move tan ak siklòn', 'kay ak konstriksyon', 'mache ak komès', 'bèt ak elvaj',
    'lapolis ak sekirite', 'lajistis ak tribinal', 'bank ak prè', 'entènèt ak rezo sosyal',
    'maladi ak remèd', 'gwosès ak tibebe', 'granmoun ak swen', 'andikap ak aksè',
    'dyaspora ak imigrasyon', 'lang ak literati', 'jounal ak radyo', 'atizana ak metye men',
    'lapèch ak lanmè', 'elektrisite ak enèji', 'fatra ak pwòpte', 'maryaj ak seremoni',
    'lantèman ak dèy', 'jaden dlo ak irigasyon', 'kredi ak sòl', 'chomaj ak djòb',
]

TASKS = [
    'yon kesyon ak repons li',
    'rezime yon ti tèks ou mete nèt andedan enstriksyon an',
    'senplifye yon fraz ki difisil pou yon moun ki pa li anpil',
    'yon vire nan yon konvèsasyon natirèl ant de moun',
    'klasifye santiman oswa entansyon yon ti mesaj ou bay',
    'rale enfòmasyon presi nan yon ti tèks ou bay',
    'reekri yon fraz pou li pi poli oswa pi kout',
    'bay yon lis etap pou fè yon bagay',
]

ANGLES = [
    'pou yon jenn moun', 'pou yon granmoun', 'nan zòn riral', 'nan yon gwo vil',
    'pou yon ti biznis', 'nan yon fanmi', 'pou yon elèv', 'pou yon moun kap travay',
    'nan yon sitiyasyon ijan', 'nan lavi chak jou', 'pou yon gwoup nan kominote a',
    'pou yon moun ki fèk kòmanse',
]

SHAPE = '[{"enstriksyon": "...", "repons": "..."}]'

TEMPLATE = '''W ap ede bati yon seri done pou antrene yon modèl lang nan kreyòl ayisyen.

Ekri {count} pè enstriksyon ak repons sou tèm sa a: {domain}, {angle}.
Kalite travay la dwe se: {task}.

Règ:
- Tout bagay dwe an kreyòl ayisyen natirèl jan yo pale l an Ayiti, ak bon òtograf ak aksan yo.
- Pa mete franse. Pa mete anglè.
- Chak enstriksyon dwe kanpe pou kont li. Si li pale sou yon tèks, mete tèks la andedan enstriksyon an.
- Chak repons dwe gen ant 2 ak 6 fraz, konkrè epi itil.
- Varye fason ou ekri yo, pa repete menm fraz la.
- Voye SÈLMAN yon JSON array, anyen dòt, konsa: ''' + SHAPE

KREYOL_WORDS = set(('yo nan pou ak se li ki gen pa mwen nou te ap sa yon lan la men epi sou tout kom '
                    'konsa fe di kap ale vin bay anpil le ou pral kote poukisa paske avek nap chak '
                    'yon ka dwe pi byen anpil').split())
FRENCH_WORDS = set(('les des est une dans pour avec sont cette nous vous leur mais plus tout aussi etre '
                    'avoir que qui par sur ses ces elle ils vos votre nos avez etes cest dont ainsi').split())
WORD = re.compile(r"[a-z']+")
META = re.compile(r'^(men yo|here are|voici|d[ak]ò|bien s|sure[,!]|of course)', re.I)


def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


def rates(text):
    words = WORD.findall(fold(text))
    if not words:
        return 0.0, 1.0
    return (sum(w in KREYOL_WORDS for w in words) / len(words),
            sum(w in FRENCH_WORDS for w in words) / len(words))


def keep(pair, floor=0.15, french_ceiling=0.05):
    """Why a pair is rejected, or None to keep it."""
    instruction, response = pair['enstriksyon'], pair['repons']
    if not (10 <= len(instruction) <= 2000):
        return 'instruction length'
    if not (40 <= len(response) <= 1500):
        return 'response length'
    if META.match(response):
        return 'teacher commentary'
    kreyol, french = rates(instruction + ' ' + response)
    if kreyol < floor:
        return 'not kreyol enough'
    if french > french_ceiling:
        return 'french'
    return None


def build(domain, task, angle, count):
    prompt = (TEMPLATE.replace('{domain}', domain).replace('{task}', task)
              .replace('{angle}', angle).replace('{count}', str(count)))
    assert '{' not in prompt.replace(SHAPE, ''), 'unreplaced field in prompt'
    return prompt


def parse(text):
    # Slice out the JSON array by hand rather than with a regex: a backslash in the source is one more
    # thing that can be mangled on the way to the pod, and find/rfind need none.
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end <= start:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [{'enstriksyon': str(r['enstriksyon']).strip(), 'repons': str(r['repons']).strip()}
            for r in rows if isinstance(r, dict) and r.get('enstriksyon') and r.get('repons')]


def main(ask=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', type=int, default=30000)
    p.add_argument('--per-call', type=int, default=10)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    p.add_argument('--model', default='google/gemma-4-26b-a4b-it')
    p.add_argument('--out', default='/workspace/sft.jsonl')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if ask is None:
        def ask(prompt):
            body = json.dumps({'model': args.model, 'temperature': 1.0, 'top_p': 0.95,
                               'max_tokens': 3000,
                               'messages': [{'role': 'user', 'content': prompt}]}).encode()
            request = urllib.request.Request(args.url, data=body,
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)['choices'][0]['message']['content']

    grid = [(d, t, a) for d in DOMAINS for t in TASKS for a in ANGLES]
    random.Random(args.seed).shuffle(grid)
    budget = -(-args.target // args.per_call) * 3      # hard ceiling on calls, whatever the yield
    print(f'{len(DOMAINS)} domains x {len(TASKS)} tasks x {len(ANGLES)} angles = {len(grid)} combinations',
          flush=True)
    print(f'aiming for {args.target:,} pairs, at most {budget:,} calls, {args.workers} at a time',
          flush=True)

    lock = threading.Lock()
    kept, seen, dropped, done = [], set(), {}, [0]
    started = time.time()

    def run(job):
        domain, task, angle = job
        if len(kept) >= args.target:
            return
        try:
            pairs = parse(ask(build(domain, task, angle, args.per_call)))
        except Exception:
            with lock:
                dropped['call failed'] = dropped.get('call failed', 0) + 1
            return
        with lock:
            for pair in pairs:
                why = keep(pair)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                    continue
                key = fold(pair['enstriksyon'])[:160]
                if key in seen:
                    dropped['duplicate'] = dropped.get('duplicate', 0) + 1
                    continue
                seen.add(key)
                pair.update(domain=domain, task=task, angle=angle)
                kept.append(pair)
            done[0] += 1
            if done[0] % 50 == 0:
                rate = len(kept) / max(time.time() - started, 1)
                left = (args.target - len(kept)) / max(rate, 0.01) / 60
                print(f'  {done[0]:,} calls, {len(kept):,} kept, {rate:.1f} pairs/s, '
                      f'{left:.0f} min to target, dropped {sum(dropped.values()):,}', flush=True)

    # Submit in waves, sizing each from the yield seen so far, so the target is actually reached
    # instead of relying on a guessed headroom factor.
    attempted = 0
    while len(kept) < args.target and attempted < budget:
        yield_per_call = (len(kept) / attempted) if attempted else args.per_call * 0.9
        need = args.target - len(kept)
        wave = min(max(int(need / max(yield_per_call, 0.5)) + 4, 8), budget - attempted)
        jobs = [grid[(attempted + i) % len(grid)] for i in range(wave)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run, job) for job in jobs]):
                future.result()
        attempted += wave

    kept = kept[:args.target]
    with open(args.out, 'w', encoding='utf-8') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
    minutes = (time.time() - started) / 60
    print('')
    print(f'kept {len(kept):,} pairs in {minutes:.1f} min -> {args.out}', flush=True)
    print('dropped: ' + json.dumps(dropped), flush=True)
    return kept, dropped


if __name__ == '__main__':
    main()
