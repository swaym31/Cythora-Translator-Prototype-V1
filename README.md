---
title: Cythora Translator
emoji: 🌍
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: other
---

# Cythora Translator — V1 Prototype

Translates English sentences into **Cythoran** (a constructed language set on the planet Cythorakh).

- **Model**: Transformer seq2seq (Encoder–Decoder)  
- **Training data**: ~5 000 English ↔ Cythoran sentence pairs  
- **Output**: Cythoran romanised script + tense/polarity analysis  
- **Use with**: [Cythora Keypad](https://swaym31.github.io/Cythora-keypad/) to render Cythoran glyphs

## How to use

1. Type an English sentence in the input box  
2. Click **Submit** — the model returns the Cythoran romanised translation  
3. Paste the romanised output into the Cythora Keypad to see the full glyph rendering  
4. Or use the **Cythora Keypad** directly — it calls this API automatically

## Uploading the model

After cloning this Space, upload `cythora_best_model.pt` to the Space root:

```bash
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/cythora-translator
cd cythora-translator
cp /path/to/cythora_best_model.pt .
git add cythora_best_model.pt
git commit -m "add model weights"
git push
```

## About Cythora

Cythora is a handcrafted constructed language with 14 clans, a custom writing system, and a full grammar rulebook. This translator is a V1 prototype — a larger model is in training.
