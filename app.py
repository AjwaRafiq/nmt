import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle

# Model Classes
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, 
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, lengths):
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.bilstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size, encoder_output_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size + encoder_output_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs, src_lengths):
        B, L, _ = encoder_outputs.size()
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).expand(B, L, -1)
        energy = self.attention(torch.cat([decoder_hidden_exp, encoder_outputs], 2)).squeeze(2)
        mask = torch.arange(L, device=encoder_outputs.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        energy = energy.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(energy, 1).masked_fill(~mask, 0)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, encoder_hidden_size, num_layers=4, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_size, encoder_hidden_size * 2)
        self.lstm = nn.LSTM(embedding_dim + encoder_hidden_size * 2, hidden_size, num_layers,
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_id, hidden, cell, encoder_outputs, src_lengths):
        embedded = self.dropout(self.embedding(input_id))
        context, _ = self.attention(hidden[-1], encoder_outputs, src_lengths)
        lstm_input = torch.cat([embedded.squeeze(1), context], 1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        return self.fc_out(output.squeeze(1)), hidden, cell

class Seq2SeqNMT(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src_ids, src_lengths, max_length=100):
        B = src_ids.size(0)
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src_ids, src_lengths)
        decoder_hidden, decoder_cell = self._init_decoder(encoder_hidden, encoder_cell, B)
        outputs = torch.zeros(B, max_length, self.decoder.vocab_size).to(self.device)
        decoder_input = torch.full((B, 1), 2, dtype=torch.long, device=self.device)
        
        for t in range(max_length):
            logits, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs, src_lengths)
            outputs[:, t, :] = logits
            decoder_input = logits.argmax(1).unsqueeze(1)
            if decoder_input.item() == 3:
                break
        return outputs
    
    def _init_decoder(self, enc_h, enc_c, B):
        enc_h = enc_h.view(self.encoder.num_layers, 2, B, self.encoder.hidden_size).mean(1)
        enc_c = enc_c.view(self.encoder.num_layers, 2, B, self.encoder.hidden_size).mean(1)
        h_list = [enc_h[i % enc_h.size(0)].unsqueeze(0) for i in range(self.decoder.num_layers)]
        c_list = [enc_c[i % enc_c.size(0)].unsqueeze(0) for i in range(self.decoder.num_layers)]
        return torch.cat(h_list, 0), torch.cat(c_list, 0)

# Tokenization Functions
def apply_bpe(term, vocab, ops):
    term = term.lower()
    units = list(term) + ['</w>']
    for a, b in ops:
        new_units, i = [], 0
        while i < len(units):
            if i < len(units)-1 and units[i]==a and units[i+1]==b:
                new_units.append(a+b)
                i += 2
            else:
                new_units.append(units[i])
                i += 1
        units = new_units
    return units

def tokenize(text, vocab, ops):
    ids = []
    for term in text.strip().lower().split():
        for sw in apply_bpe(term, vocab, ops):
            ids.append(vocab.get(sw, vocab.get('<UNK>', 1)))
    return ids

def detokenize(ids, vocab):
    rev = {v: k for k, v in vocab.items()}
    tokens = [rev.get(i, '') for i in ids if i not in [0,1,2,3]]
    return ''.join(tokens).replace('</w>', ' ').strip()

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    
    with open('urdu_tokenizer.pkl', 'rb') as f:
        ur = pickle.load(f)
    with open('roman_tokenizer.pkl', 'rb') as f:
        ro = pickle.load(f)
    
    encoder = Encoder(560, 256, 256, 2, 0.3)
    decoder = Decoder(534, 256, 256, 256, 4, 0.3)
    model = Seq2SeqNMT(encoder, decoder, device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    return model, ur['token_map'], ur['merge_history'], ro['token_map'], ro['merge_history'], device

def translate(text, model, ur_vocab, ur_ops, ro_vocab, ro_ops, device):
    ids = tokenize(text, ur_vocab, ur_ops)
    if not ids:
        return "Invalid input"
    
    src = torch.tensor([ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(src, lengths)
    
    pred_ids = outputs.argmax(dim=-1)[0].cpu().numpy()
    return detokenize(pred_ids, ro_vocab)

# Streamlit App
st.set_page_config(page_title="Urdu to Roman", page_icon="🔤", layout="wide")
st.title("🔤 Urdu to Roman Transliteration")
st.markdown("### Convert Urdu script to Roman/Latin script")

try:
    model, ur_vocab, ur_ops, ro_vocab, ro_ops, device = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input (Urdu)")
    urdu = st.text_area("Enter Urdu text:", height=200, placeholder="اردو متن یہاں لکھیں...")
    btn = st.button("🔄 Translate", type="primary", use_container_width=True)

with col2:
    st.subheader("📤 Output (Roman)")
    if btn and urdu.strip():
        with st.spinner("Translating..."):
            try:
                result = translate(urdu, model, ur_vocab, ur_ops, ro_vocab, ro_ops, device)
                st.text_area("Translation:", value=result, height=200, disabled=True)
                st.code(result, language=None)
            except Exception as e:
                st.error(f"Translation error: {e}")
    elif btn:
        st.warning("⚠️ Please enter Urdu text first!")

st.markdown("---")
st.markdown("### 📚 Example Translations")

examples = [
    ("آنکھ سے دور نہ ہو دل سے اتر جائے گا", "Example 1"),
    ("دل سے تیری نگاہ جگر تک اتر گئی", "Example 2"),
    ("تو کبھی خود کو بھی دیکھے گا تو ڈر جائے گا", "Example 3"),
]

cols = st.columns(3)
for idx, (example, label) in enumerate(examples):
    with cols[idx]:
        if st.button(f"Try {label}", use_container_width=True):
            st.session_state['example'] = example

st.markdown("---")
st.markdown("""
### ℹ️ About
**Model:** Seq2Seq with Attention (BiLSTM Encoder + LSTM Decoder)  
**Training Data:** 21,003 Urdu poetry verses from 30 classical poets  
**Tokenization:** Byte Pair Encoding (BPE)
""")