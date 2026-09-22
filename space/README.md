---
title: Makandal Base
emoji: 📜
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Test the Kreyol base model while it trains
---

# Makandal — modèl baz kreyòl

A 111M-parameter Haitian Creole base model, trained from scratch on 466M deduplicated Kreyòl tokens
(FineWeb-2, HPLT, MADLAD-400, finepdfs, Wikipedia, GlotCC and Glot500) with a 32,768-token Kreyòl BPE.

This Space loads the newest training checkpoint from `jsbeaudry/makandal-base`, so the model behind it
improves while the run continues. The header shows which step you are talking to.

It is a **base** model: it continues text. It is not instruction-tuned and will not answer questions.

## Setup

The model repository is private, so this Space needs an `HF_TOKEN` secret with read access to it
(Settings → Variables and secrets → New secret). Without it the Space cannot download the checkpoint.
