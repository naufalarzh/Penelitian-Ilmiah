import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ====================
# Konfigurasi Perangkat
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# Fungsi Load Model dan Tokenizer dari Hugging Face
# ====================
@st.cache_resource
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model

# ====================
# Fungsi Load LabelEncoder dari Hugging Face
# ====================
@st.cache_resource
def load_label_encoder_from_hf(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# ====================
# Load Semua Komponen
# ====================
tokenizer_topic, model_topic = load_model_and_tokenizer("arkan03/indobert-topic")
tokenizer_sentiment, model_sentiment = load_model_and_tokenizer("arkan03/indobert-sentiment")

topic_encoder = load_label_encoder_from_hf("arkan03/indobert-topic", "topic_encoder.pkl")
sentiment_encoder = load_label_encoder_from_hf("arkan03/indobert-sentiment", "sentiment_encoder.pkl")

# ====================
# Fungsi Prediksi
# ====================
def predict(text):
    # Tokenisasi untuk model topik
    inputs_topic = tokenizer_topic(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs_topic = {k: v.to(device) for k, v in inputs_topic.items()}
    with torch.no_grad():
        logits_topic = model_topic(**inputs_topic).logits
        probs_topic = F.softmax(logits_topic, dim=1).cpu().numpy()[0]
        topic_pred = topic_encoder.inverse_transform([np.argmax(probs_topic)])[0]

    # Tokenisasi untuk model sentimen
    inputs_sentiment = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs_sentiment = {k: v.to(device) for k, v in inputs_sentiment.items()}
    with torch.no_grad():
        logits_sent = model_sentiment(**inputs_sentiment).logits
        probs_sent = F.softmax(logits_sent, dim=1).cpu().numpy()[0]
        sentiment_pred = sentiment_encoder.inverse_transform([np.argmax(probs_sent)])[0]

    return topic_pred, sentiment_pred, probs_topic, probs_sent

# ====================
# STREAMLIT UI
# ====================
st.set_page_config(page_title="Klasifikasi Topik & Sentimen", layout="centered")
# Judul
st.markdown("## 📊 Dashboard Klasifikasi Topik & Sentimen")

# Instruksi
st.markdown("Masukkan kalimat di bawah ini untuk memprediksi topik dan sentimennya.")
st.markdown("_kalimat yang dimasukkan sebaiknya mengandung topik **ekonomi**, **politik**, atau **pendidikan**._")

# Input Teks
text = st.text_area("Masukkan Kalimat:", height=150)

# Tombol Prediksi
if st.button("Prediksi"):
    if text.strip():
        try:
            topic, sentiment, probs_topic, probs_sentiment = predict(text)

            st.success(f"✅ **Prediksi: Sentimen _{sentiment.upper()}_ pada topik _{topic.upper()}_**")

            st.markdown("### Probabilitas Topik:")
            for label, prob in zip(topic_encoder.classes_, probs_topic):
                st.markdown(f"- **{label.capitalize()}** : **{prob*100:.2f}%**")

            st.markdown("### Probabilitas Sentimen:")
            for label, prob in zip(sentiment_encoder.classes_, probs_sentiment):
                st.markdown(f"- **{label.capitalize()}** : **{prob*100:.2f}%**")

        except Exception as e:
            st.error(f"❌ Terjadi error saat memproses prediksi: {e}")
    else:
        st.warning("⚠️ Teks tidak boleh kosong.")

st.markdown("---")
st.caption("Penelitian Ilmiah, Naufal Arkan Zhafran 2025.")
