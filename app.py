# ══════════════════════════════════════════════════════════════════
#  Cythora Translator — HuggingFace Spaces Backend
#  Model: Transformer seq2seq (English → Cythoran romanised)
#  Architecture and weights: cythora_best_model.pt
# ══════════════════════════════════════════════════════════════════

import re
import math
import json
import torch
import torch.nn as nn
import gradio as gr

# ── Device ────────────────────────────────────────────────────────
device = torch.device('cpu')   # Spaces free tier — CPU only

# ═══════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE  (mirrors Cythora_translator.ipynb Cell 9)
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads,
                 n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_key_padding_mask):
        x = self.embed(src) * self.scale
        x = self.pos_enc(x)
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads,
                 n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale  = math.sqrt(d_model)

    def forward(self, tgt, memory, tgt_mask,
                tgt_key_padding_mask, memory_key_padding_mask):
        x = self.embed(tgt) * self.scale
        x = self.pos_enc(x)
        x = self.transformer_decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(x)


class CythoraTranslator(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, d_model,
                 n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.encoder = Encoder(enc_vocab, d_model, n_heads, n_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(dec_vocab, d_model, n_heads, n_layers, d_ff, dropout, max_len)

    def make_src_mask(self, src):
        return (src == 0)

    def make_tgt_mask(self, tgt):
        T    = tgt.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask.to(tgt.device)

    def forward(self, src, tgt):
        src_mask     = self.make_src_mask(src)
        tgt_mask     = self.make_tgt_mask(tgt)
        tgt_pad_mask = self.make_src_mask(tgt)
        memory       = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask, tgt_pad_mask, src_mask)


# ═══════════════════════════════════════════════════════════════════
#  TOKENIZER  (mirrors Cell 7)
# ═══════════════════════════════════════════════════════════════════

PAD = '<pad>'; SOS = '<sos>'; EOS = '<eos>'; UNK = '<unk>'

def tokenize(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,'])", r" \1 ", sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence.split()

def encode(sentence, word2idx, max_len=100):
    tokens = tokenize(sentence)
    ids    = [word2idx[SOS]]
    for tok in tokens[:max_len - 2]:
        ids.append(word2idx.get(tok, word2idx[UNK]))
    ids.append(word2idx[EOS])
    ids += [word2idx[PAD]] * (max_len - len(ids))
    return ids

def decode_ids(ids, idx2word):
    tokens = []
    for i in ids:
        word = idx2word.get(i, UNK)
        if word == EOS:
            break
        if word not in [PAD, SOS]:
            tokens.append(word)
    return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════════════
#  LANGUAGE CONSTANTS  (from Cell 3 & 5)
# ═══════════════════════════════════════════════════════════════════

NEGATION = 'khed'

PRONOUN_MAP = {
    'i':'oiva','you':'vat','he':'ret','she':'rat',
    'we':'jet','they':'jat','it':'ova',
    'me':'osi','him':'resi','her':'rasi','us':'jesi',
    'them':'jasi','this':'irov','that':'orov',
    'these':'jirov','those':'jorov',
    'my':'oivam','your':'vatam','his':'retam','hers':'ratam',
    'our':'jetam','their':'jatam','its':'ovam',
    'myself':'osio','yourself':'vasio','himself':'resio',
    'herself':'rasio','ourselves':'jesio','themselves':'jasio',
}

# ═══════════════════════════════════════════════════════════════════
#  TENSE DETECTION  (from Cell 5)
# ═══════════════════════════════════════════════════════════════════

def detect_tense(sentence):
    s     = sentence.lower().strip()
    words = s.split()

    negative = any(w in s for w in ["n't","not","never","no longer","neither","nobody","nothing"])

    subject = None
    for w in words[:3]:
        clean = w.strip("?,.'\"")
        if clean in PRONOUN_MAP:
            subject = clean
            break

    passive_pattern = re.compile(
        r'\b(am|is|are|was|were)\s+being\s+\w+(?<!ing)\b|'
        r'\b(am|is|are|was|were)\s+(?!being)\w+(?:ed|en|t|wn)(?!ing)\b|'
        r'\b(will|would|could|should|shall|may|might|must)\s+be\s+\w+(?<!ing)\b|'
        r'\bby\s+(me|you|him|her|us|them|it)\b'
    )
    passive = bool(passive_pattern.search(s))

    words_clean = [w.strip(".,?!'\"") for w in words]
    if any(w.endswith('ing') for w in words_clean[-3:]):
        if not re.search(r'\bby\s+(me|you|him|her|us|them|it)\b', s):
            passive = False
    if re.search(r'\bby\s+(me|you|him|her|us|them|it)\b', s):
        passive = True

    aux_list = [
        'will have been','would have been','could have been',
        'should have been','shall have been','may have been',
        'might have been','must have been',
        'will have','would have','could have','should have',
        'shall have','may have','might have','must have',
        'will be','would be','could be','should be',
        'shall be','may be','might be','must be',
        'have been','has been','had been',
        'will','would','could','should','shall',
        'may','might','must','can',
        'have to','has to','had to',
        'keep','keeps','kept','will keep',
        'used to','just',
        'have','has','had',
        'am','is','are','was','were',
    ]
    aux = None
    for a in aux_list:
        if re.search(r'\b' + re.escape(a) + r'\b', s):
            aux = a; break

    tense = 'present_simple'

    if passive:
        if aux == 'will have been':         tense = 'passive_future_perfect'
        elif aux == 'would have been':      tense = 'passive_would_perfect'
        elif aux == 'could have been':      tense = 'passive_could_perfect'
        elif aux == 'should have been':     tense = 'passive_should_perfect'
        elif aux == 'will be':              tense = 'passive_future_simple'
        elif aux in ('have been','has been'): tense = 'passive_present_perfect'
        elif aux == 'had been':             tense = 'passive_past_perfect'
        elif aux in ('was','were'):         tense = 'passive_past_simple'
        elif aux in ('am','is','are'):      tense = 'passive_present_simple'
        else:                               tense = 'passive_present_simple'
    elif aux == 'will have been':           tense = 'future_perfect_continuous'
    elif aux == 'will have':                tense = 'future_perfect'
    elif aux == 'would have':               tense = 'would_perfect'
    elif aux == 'could have':               tense = 'could_perfect'
    elif aux == 'should have':              tense = 'should_perfect'
    elif aux == 'will be':                  tense = 'future_continuous'
    elif aux == 'would be':                 tense = 'would_continuous'
    elif aux == 'could be':                 tense = 'could_continuous'
    elif aux == 'should be':               tense = 'should_continuous'
    elif aux == 'will':                    tense = 'future_simple'
    elif aux == 'would':                   tense = 'would_simple'
    elif aux == 'could':                   tense = 'could_simple'
    elif aux == 'should':                  tense = 'should_simple'
    elif aux == 'shall':                   tense = 'shall_simple'
    elif aux == 'may':                     tense = 'may_simple'
    elif aux == 'might':                   tense = 'might_simple'
    elif aux == 'must':                    tense = 'must_simple'
    elif aux == 'can':                     tense = 'can'
    elif aux in ('have to','has to'):      tense = 'have_to'
    elif aux == 'had to':                  tense = 'had_to'
    elif aux in ('keep','keeps'):          tense = 'keep_present'
    elif aux == 'kept':                    tense = 'keep_past'
    elif aux == 'used to':                 tense = 'used_to'
    elif aux in ('have been','has been'):  tense = 'present_perfect_continuous'
    elif aux == 'had been':               tense = 'past_perfect_continuous'
    elif aux in ('have','has'):           tense = 'present_perfect'
    elif aux == 'had':                    tense = 'past_perfect'
    elif aux in ('am','is','are'):        tense = 'present_continuous'
    elif aux in ('was','were'):           tense = 'past_continuous'
    else:
        tense = 'past_simple' if (words and words[-1].endswith('ed')) else 'present_simple'

    return {'tense': tense, 'negative': negative,
            'passive': passive, 'subject': subject, 'aux': aux}


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

try:
    checkpoint = torch.load('cythora_best_model.pt', map_location=device)
    ENG_W2I    = checkpoint['eng_w2i']
    CYTH_W2I   = checkpoint['cyth_w2i']
    CYTH_I2W   = {int(k): v for k, v in checkpoint['cyth_i2w'].items()} \
                 if isinstance(list(checkpoint['cyth_i2w'].keys())[0], str) \
                 else checkpoint['cyth_i2w']

    model = CythoraTranslator(
        enc_vocab = len(ENG_W2I),
        dec_vocab = len(CYTH_W2I),
        d_model   = 256,
        n_heads   = 8,
        n_layers  = 3,
        d_ff      = 512,
        dropout   = 0.3,
        max_len   = 100,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    MODEL_LOADED = True
    print(f"✓ Model loaded — epoch {checkpoint.get('epoch','?')}, "
          f"val loss {checkpoint.get('val_loss', '?'):.4f}")
except FileNotFoundError:
    MODEL_LOADED = False
    print("⚠ cythora_best_model.pt not found — upload the file to this Space.")


# ═══════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════

def translate(sentence, max_len=100):
    if not MODEL_LOADED:
        return "[model not loaded]"
    model.eval()
    with torch.no_grad():
        src     = encode(sentence, ENG_W2I, max_len)
        src     = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
        sos_idx = CYTH_W2I[SOS]
        eos_idx = CYTH_W2I[EOS]
        dec_in  = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
        tokens  = []
        for _ in range(max_len):
            out        = model(src, dec_in)
            next_tok   = out[:, -1, :].argmax(dim=-1).item()
            if next_tok == eos_idx:
                break
            tokens.append(next_tok)
            dec_in = torch.cat([dec_in,
                                 torch.tensor([[next_tok]], dtype=torch.long).to(device)], dim=1)
        return decode_ids(tokens, CYTH_I2W)


def translate_full(sentence):
    analysis      = detect_tense(sentence)
    neural_output = translate(sentence)
    if analysis['negative'] and NEGATION not in neural_output:
        neural_output = neural_output + f' {NEGATION}'
    return {
        'romanised': neural_output,
        'tense':     analysis['tense'],
        'negative':  analysis['negative'],
        'passive':   analysis['passive'],
        'subject':   analysis['subject'],
    }


# ═══════════════════════════════════════════════════════════════════
#  GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════

def predict(english_text: str):
    english_text = english_text.strip()
    if not english_text:
        return "", "—", "—", "—"
    result = translate_full(english_text)
    tense_label = result['tense'].replace('_', ' ')
    return (
        result['romanised'],
        tense_label,
        "yes" if result['negative'] else "no",
        "yes" if result['passive']  else "no",
    )


demo = gr.Interface(
    fn          = predict,
    inputs      = gr.Textbox(label="English sentence", placeholder="e.g. She will not help us."),
    outputs     = [
        gr.Textbox(label="Cythoran  (romanised)"),
        gr.Textbox(label="Tense detected"),
        gr.Textbox(label="Negative"),
        gr.Textbox(label="Passive voice"),
    ],
    title       = "Cythora Translator — V1 Prototype",
    description = (
        "Translates English sentences into Cythoran romanised script. "
        "Trained on 5 000 sentence pairs. "
        "Feed the romanised output into the Cythora Keypad to render glyphs."
    ),
    examples    = [
        ["I walk to the river."],
        ["She is very kind."],
        ["They will help us."],
        ["He did not know that."],
        ["We are walking together."],
    ],
    api_name    = "predict",
    allow_flagging = "never",
)

if __name__ == "__main__":
    demo.launch()
