---
title: Makandal Instruct
emoji: 🗣️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Kreyol instruction model that knows when to stop
---

# Makandal — modèl kreyòl ki swiv enstriksyon

The 111M-parameter Haitian Creole base model (`jsbeaudry/makandal-base`), instruction-tuned on 30,000
Kreyòl instruction and response pairs across 40 domains and 8 task types. The loss covered the response
only, so the model learned to answer rather than to write instructions.

Measured on held-out instructions it never saw: response loss 3.090 → 1.965, next-token accuracy
35.7% → 53.8%, and it ends its own answer on 20 of 20 held-out prompts.

**What it is not.** Tuning taught the shape of a good answer, not facts. At this size it can state
something false fluently — asked for the capital of Haiti, it wrote about the economy. It is most
reliable at naming the tone of a message and rewriting a sentence more politely. Summaries and extracting
details from a text are hit or miss: in testing it summarised a flood report by inventing a water
shortage, and pulled the date out of a meeting notice but dropped the place.

## Setup

The model repository is private, so this Space needs an `HF_TOKEN` secret with read access to it
(Settings → Variables and secrets → New secret). Without it the Space cannot download the model.

The prompt template is read from the `sft_summary.json` the trainer wrote, so this page always prompts
the model exactly as it was trained.
