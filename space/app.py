"""Makandal base model: the finished 111M Kreyol model, loaded from the root of jsbeaudry/makandal-base.

Training is over, so this serves the final weights (the checkpoint with the lowest held-out loss) rather
than the mid-run checkpoints the earlier version of this Space polled for.
"""
import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get('MAKANDAL_REPO', 'jsbeaudry/makandal-base')
TOKEN = os.environ.get('HF_TOKEN')
state = {'model': None, 'tokenizer': None}


def measured():
    """Numbers from the run itself, so the page cannot drift from what was actually measured."""
    lines = []
    try:
        summary = json.load(open(hf_hub_download(REPO, 'training_summary.json', token=TOKEN)))
        lines.append('**{:.0f}M parameters** trained on **{:,.0f}M tokens** over {:g} passes'.format(
            summary['parameters'] / 1e6, summary['unique_train_tokens'] / 1e6, summary['epochs']))
        lines.append('held-out perplexity **{:,.1f}**'.format(summary['best_eval_perplexity']))
    except Exception:
        pass
    try:
        both = json.load(open(hf_hub_download(REPO, 'comparison.json', token=TOKEN)))
        new = both.get('jsbeaudry/makandal-base', {}).get('bits_per_character')
        old = both.get('jsbeaudry/makandal-pre-trained', {}).get('bits_per_character')
        if new and old:
            lines.append('**{:.3f} bits per character** on held-out Kreyol, against {:.3f} for the earlier '
                         'makandal-pre-trained ({:.1f} times fewer bits for the same text)'.format(
                             new, old, old / new))
    except Exception:
        pass
    return ' &nbsp;|&nbsp; '.join(lines) if lines else 'Final model loaded.'


def load():
    state['tokenizer'] = AutoTokenizer.from_pretrained(REPO, token=TOKEN)
    state['model'] = AutoModelForCausalLM.from_pretrained(REPO, token=TOKEN, dtype=torch.float32).eval()
    return measured()


def write(prompt, new_tokens, temperature, top_p, repetition_penalty):
    if state['model'] is None:
        load()
    if not prompt.strip():
        return 'Ekri yon bagay anvan. (Write something first.)'
    tok, model = state['tokenizer'], state['model']
    ids = tok(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=int(new_tokens), do_sample=temperature > 0,
                             temperature=max(temperature, 0.01), top_p=top_p,
                             repetition_penalty=repetition_penalty,
                             pad_token_id=tok.pad_token_id or 1, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


try:
    READY = load()
except Exception as error:
    READY = ('**Could not load the model.** {}\n\n`{}` is private, so this Space needs an `HF_TOKEN` '
             'secret with read access (Settings, then Variables and secrets).'.format(str(error)[:200], REPO))

with gr.Blocks(title='Makandal base') as demo:
    gr.Markdown('# Makandal, modèl baz kreyòl\n'
                'Yon modèl lang kreyòl ayisyen ki antrene depi zewo. '
                'Li kontinye tèks: li pa yon chatbot, li pa reponn kesyon.\n\n'
                '*A Haitian Creole base model trained from scratch. It continues text; it is not '
                'instruction-tuned and will not answer questions.*')
    gr.Markdown(READY)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label='Kòmansman tèks la / Prompt', lines=3, value='Ayiti se yon peyi ki')
            go = gr.Button('Ekri / Generate', variant='primary')
            output = gr.Textbox(label='Rezilta / Output', lines=12)
        with gr.Column(scale=1):
            new_tokens = gr.Slider(16, 400, value=150, step=8, label='Longè / Length')
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label='Temperature')
            top_p = gr.Slider(0.1, 1.0, value=0.92, step=0.01, label='Top-p')
            penalty = gr.Slider(1.0, 1.5, value=1.1, step=0.01, label='Repetition penalty')
    gr.Examples([['Ayiti se yon peyi ki'], ['Lekòl la te louvri'], ['Nan maten an, mwen'],
                 ['Istwa Ayiti kòmanse'], ['Manje kreyòl la gen'], ['Doktè a di mwen']], inputs=prompt)
    go.click(write, [prompt, new_tokens, temperature, top_p, penalty], output)

if __name__ == '__main__':
    demo.launch()
