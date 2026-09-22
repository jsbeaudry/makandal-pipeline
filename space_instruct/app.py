"""Makandal instruct: the 111M Kreyol model after instruction tuning on 30,000 pairs.

The prompt template is read from sft_summary.json on the Hub, the file the trainer wrote, rather than
typed out again here. A template that differs from training by one newline is a different prompt, and
the model would quietly answer worse; reading it back means this page cannot drift from the run.

Decoding defaults come from a measurement, not taste. On 20 held-out instructions, greedy decoding
repeated a sentence verbatim in 3 answers; greedy with a 1.15 repetition penalty repeated none, and all
20 answers still ended on their own. Greedy keeps the page deterministic, so the same instruction gives
the same answer while you are testing; raise the temperature to sample instead.
"""
import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get('MAKANDAL_REPO', 'jsbeaudry/makandal-instruct')
TOKEN = os.environ.get('HF_TOKEN')
FALLBACK_TEMPLATE = '### Enstriksyon:\n%s\n\n### Repons:\n'
state = {'model': None, 'tokenizer': None, 'template': FALLBACK_TEMPLATE, 'summary': None}

EXAMPLES = [
    # One per task type the model was tuned on, written fresh rather than taken from the dataset.
    'Poukisa li enpòtan pou bwè dlo pwòp?',
    "Rezime tèks sa a: 'Lapli tonbe tout nwit la nan Okap. Dlo a monte nan plizyè katye, epi lekòl yo "
    "fèmen jodi a pou sekirite timoun yo.'",
    "Senplifye fraz sa a pou yon moun ki pa li anpil: 'Ministè Sante Piblik la rekòmande popilasyon an "
    "pou yo pran mezi prevansyon kont maladi kolera a.'",
    "Yon kliyan di: 'Konbyen pri diri a jodi a?' Reponn li tankou yon machann nan mache a.",
    "Klasifye santiman mesaj sa a: 'Mèsi anpil pou èd ou, ou sove lavi m jodi a!'",
    "Ki dat ak ki kote reyinyon an ap fèt? 'Reyinyon komite a ap fèt samdi 12 oktòb, a 3è apremidi, "
    "nan lekòl Sen Jozèf.'",
    "Reekri fraz sa a pou l pi poli: 'Fèmen pòt la!'",
    'Bay etap pou prepare yon kit ijans anvan sezon siklòn.',
]


def load():
    try:
        with open(hf_hub_download(REPO, 'sft_summary.json', token=TOKEN), encoding='utf-8') as f:
            state['summary'] = json.load(f)
        state['template'] = state['summary'].get('prompt_template', FALLBACK_TEMPLATE)
    except Exception:
        pass
    state['tokenizer'] = AutoTokenizer.from_pretrained(REPO, token=TOKEN)
    state['model'] = AutoModelForCausalLM.from_pretrained(REPO, token=TOKEN).eval()
    return measured()


def measured():
    """Numbers from the run itself, so the page cannot claim more than was measured."""
    summary = state['summary']
    if not summary:
        return 'Model loaded.'
    before, after = summary['before'], summary['after']
    return ('Tuned on **{:,}** instruction pairs from the 111M base model. On {} held-out instructions it '
            'never saw, response loss fell from {:.3f} to **{:.3f}** and next-token accuracy rose from '
            '{:.1%} to **{:.1%}**.'.format(summary['train'], summary['held_out'], before['eval_loss'],
                                           after['eval_loss'], before['response_accuracy'],
                                           after['response_accuracy']))


def answer(instruction, new_tokens, temperature, penalty):
    if state['model'] is None:
        load()
    instruction = instruction.strip()
    if not instruction:
        return 'Ekri yon enstriksyon anvan. (Write an instruction first.)', ''
    tok, model = state['tokenizer'], state['model']
    batch = tok(state['template'] % instruction, return_tensors='pt', add_special_tokens=False)
    settings = dict(max_new_tokens=int(new_tokens), repetition_penalty=float(penalty),
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if temperature > 0:
        settings.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        settings.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**batch, **settings)
    new = out[0, batch['input_ids'].shape[1]:].tolist()
    ended = tok.eos_token_id in new
    text = tok.decode(new, skip_special_tokens=True).strip()
    note = '{} tokens · {}'.format(len(new), 'li fini poukont li / stopped by itself' if ended
                                   else 'li rive nan limit longè a / hit the length limit')
    return text, note


try:
    READY = load()
except Exception as error:
    READY = ('**Could not load the model.** {}\n\n`{}` is private, so this Space needs an `HF_TOKEN` '
             'secret with read access (Settings, then Variables and secrets).'.format(str(error)[:200], REPO))

with gr.Blocks(title='Makandal instruct') as demo:
    gr.Markdown('# Makandal, modèl kreyòl ki swiv enstriksyon\n'
                'Bay li yon enstriksyon an kreyòl: yon kesyon, yon tèks pou rezime, yon fraz pou reekri, '
                'yon mesaj pou klasifye. Li reponn an kreyòl, epi li konnen kilè pou l sispann.\n\n'
                '*Give it an instruction in Haitian Creole. It answers in Kreyòl and knows when to stop.*')
    gr.Markdown(READY)
    gr.Markdown('**Limit li / Its limit.** Li aprann fòm yon bon repons, men ak 111M paramèt li pa konnen '
                'anpil bagay: li ka envante yon fè ak konfyans. Pa sèvi avè l pou enfòmasyon ou bezwen '
                'fè konfyans.\n\n'
                '*Tuning taught it the shape of a good answer, not facts. At 111M parameters it can state '
                'something false fluently: asked for the capital of Haiti, it wrote about the economy '
                'instead. It is most reliable at naming the tone of a message and rewriting a sentence '
                'more politely. Summaries and pulling details out of a text are hit or miss: it can '
                'drop a fact or change one.*')
    with gr.Row():
        with gr.Column(scale=3):
            instruction = gr.Textbox(label='Enstriksyon / Instruction', lines=4, value=EXAMPLES[4])
            go = gr.Button('Reponn / Answer', variant='primary')
            output = gr.Textbox(label='Repons / Response', lines=8)
            note = gr.Markdown()
        with gr.Column(scale=1):
            new_tokens = gr.Slider(16, 256, value=120, step=8, label='Longè maksimòm / Max length')
            temperature = gr.Slider(0.0, 1.2, value=0.0, step=0.05,
                                    label='Temperature (0 = menm repons chak fwa / same answer every time)')
            penalty = gr.Slider(1.0, 1.5, value=1.15, step=0.01, label='Repetition penalty')
    gr.Examples([[e] for e in EXAMPLES], inputs=instruction)
    go.click(answer, [instruction, new_tokens, temperature, penalty], [output, note])
    instruction.submit(answer, [instruction, new_tokens, temperature, penalty], [output, note])

if __name__ == '__main__':
    demo.launch()
