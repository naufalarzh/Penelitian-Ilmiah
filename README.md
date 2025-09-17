# 📊 Dashboard Klasifikasi Topik & Sentimen

Proyek ini adalah aplikasi berbasis **Streamlit** untuk **memprediksi topik dan sentimen** dari sebuah kalimat berbahasa Indonesia.  
Model yang digunakan adalah fine-tuned IndoBERT yang di-host di Hugging Face.

---

## 🚀 Fitur

- Klasifikasi **Topik**: `ekonomi`, `politik`, `pendidikan`
- Klasifikasi **Sentimen**: `positif`, `netral`, `negatif`
- Menampilkan **probabilitas** setiap kelas
- UI interaktif berbasis **Streamlit**

---

## 🔧 Teknologi

- Python 3.11+
- Streamlit
- Transformers (Hugging Face)
- PyTorch
- Scikit-learn
- huggingface_hub
- NumPy

---

## ⚙️ Model yang Digunakan

- **Model Topik:** [`arkan03/indobert-topic`](https://huggingface.co/arkan03/indobert-topic)
- **Model Sentimen:** [`arkan03/indobert-sentiment`](https://huggingface.co/arkan03/indobert-sentiment)

Label encoder (**`topic_encoder.pkl`** dan **`sentiment_encoder.pkl`**) juga diambil dari repository yang sama di Hugging Face Hub.


---
## 📥 Instalasi

1. Clone repository ini:

   ```bash
   git clone https://github.com/naufalarzh/Penelitian-Ilmiah
   cd Penelitian-Ilmiah
   code .
    ```
2. Buat Virtual Environment (Opsional tapi Disarankan)

    ```bash
    python -m venv .venv
    == cara mengaktifkan ==
    .venv\Scripts\activate (Windows (PowerShell / CMD))
    source .venv/bin/activate (macOS / Linux )
    source .venv/Scripts/activate (Bash)
    ```

3. Install Requirements
    ```bash
       pip install -r requirements.txt
    ```   
4. Jalankan Program
    ```bash
       streamlit run app.py

    ```   
