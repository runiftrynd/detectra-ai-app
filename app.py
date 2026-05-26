# ============================================================
# DETECTRA AI
# Aplikasi Deteksi Teks AI dan Manusia Menggunakan BERT
# ============================================================

import re
import time
from typing import Dict

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Detectra AI | Deteksi Teks AI dan Manusia",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# KONFIGURASI MODEL
# ============================================================

# Repository model Hugging Face milik Anda
MODEL_REPO = "runiftrynd/detectra-ai"

# Sama dengan max_length pada saat training BERT
MAX_LENGTH = 128

# Label sesuai penelitian
LABEL_MAP = {
    0: "Teks Manusia",
    1: "Teks AI"
}


# ============================================================
# CSS TAMPILAN
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-section {
            padding: 2.8rem 2.5rem;
            border-radius: 26px;
            background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 70%, #2563eb 100%);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
            margin-bottom: 2rem;
        }

        .hero-badge {
            display: inline-block;
            background-color: rgba(255,255,255,0.15);
            color: #dbeafe;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .hero-title {
            color: white;
            font-size: 3rem;
            font-weight: 750;
            margin-bottom: 0.7rem;
        }

        .hero-description {
            color: #dbeafe;
            font-size: 1.05rem;
            max-width: 750px;
            line-height: 1.7;
        }

        .step-card {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.1rem;
            margin-bottom: 0.9rem;
        }

        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #dbeafe;
            color: #1d4ed8;
            border-radius: 8px;
            width: 28px;
            height: 28px;
            font-weight: bold;
            margin-right: 0.5rem;
        }

        .step-title {
            color: #0f172a;
            font-weight: 650;
        }

        .step-description {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.55rem;
        }

        .result-ai {
            background-color: #fff1f2;
            border: 1px solid #fecdd3;
            border-radius: 18px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .result-human {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .result-title-ai {
            color: #be123c;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .result-title-human {
            color: #1d4ed8;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .result-description {
            color: #334155;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }

        .confidence {
            color: #0f172a;
            font-weight: 650;
        }

        .warning-note {
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            color: #713f12;
            padding: 1rem;
            border-radius: 14px;
            margin-top: 1rem;
            line-height: 1.6;
            font-size: 0.9rem;
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            padding-top: 3rem;
        }

        div.stButton > button {
            width: 100%;
            height: 3rem;
            border-radius: 12px;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEMUAT MODEL DARI HUGGING FACE HUB
# ============================================================

@st.cache_resource(show_spinner="Memuat model Detectra AI...")
def load_model():
    """
    Memuat tokenizer dan model BERT dari Hugging Face Hub.
    Cache digunakan agar model tidak dimuat ulang setiap tombol diklik.
    """

    # Jika repository Hugging Face public, token tidak diperlukan.
    # Jika private, HF_TOKEN dapat ditambahkan melalui Streamlit Secrets.
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=hf_token
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        token=hf_token
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


try:
    tokenizer, model, device = load_model()

except Exception as error:
    st.error(
        "Model tidak berhasil dimuat. Pastikan repository Hugging Face "
        "`runiftrynd/detectra-ai-bert` sudah tersedia dan memiliki file model."
    )

    st.code(str(error))

    st.info(
        "Apabila repository Hugging Face Anda private, tambahkan HF_TOKEN "
        "pada pengaturan Secrets di Streamlit."
    )

    st.stop()


# ============================================================
# FUNGSI PENDUKUNG
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalisasi ringan input pengguna.
    Tanda baca tidak dihapus karena BERT menggunakan konteks teks asli.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def count_words(text: str) -> int:
    """
    Menghitung jumlah kata pada input.
    """
    if not text.strip():
        return 0

    return len(text.split())


def predict_text(text: str) -> Dict[str, float]:
    """
    Melakukan prediksi teks menggunakan model BERT.
    """

    processed_text = normalize_text(text)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    probability_human = float(probabilities[0].cpu().item())
    probability_ai = float(probabilities[1].cpu().item())

    label_id = int(torch.argmax(probabilities).cpu().item())
    confidence = float(probabilities[label_id].cpu().item())

    return {
        "label_id": label_id,
        "label": LABEL_MAP[label_id],
        "confidence": confidence,
        "probability_human": probability_human,
        "probability_ai": probability_ai
    }


def display_result(result: Dict[str, float]) -> None:
    """
    Menampilkan hasil klasifikasi ke halaman aplikasi.
    """

    confidence_percent = result["confidence"] * 100
    human_percent = result["probability_human"] * 100
    ai_percent = result["probability_ai"] * 100

    if result["label_id"] == 1:
        st.markdown(
            f"""
            <div class="result-ai">
                <div class="result-title-ai">
                    Terdeteksi sebagai Teks AI
                </div>
                <div class="result-description">
                    Model memperkirakan bahwa teks memiliki pola bahasa
                    yang lebih dekat dengan teks hasil generasi AI.
                </div>
                <div class="confidence">
                    Tingkat keyakinan model: {confidence_percent:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="result-human">
                <div class="result-title-human">
                    Terdeteksi sebagai Teks Manusia
                </div>
                <div class="result-description">
                    Model memperkirakan bahwa teks memiliki pola bahasa
                    yang lebih dekat dengan tulisan manusia.
                </div>
                <div class="confidence">
                    Tingkat keyakinan model: {confidence_percent:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilitas Teks Manusia",
            f"{human_percent:.2f}%"
        )
        st.progress(result["probability_human"])

    with col2:
        st.metric(
            "Probabilitas Teks AI",
            f"{ai_percent:.2f}%"
        )
        st.progress(result["probability_ai"])

    st.markdown(
        """
        <div class="warning-note">
            <strong>Catatan:</strong> Hasil ini merupakan prediksi model
            machine learning dan tidak dapat digunakan sebagai satu-satunya
            dasar untuk menentukan orisinalitas, plagiarisme, atau
            pelanggaran akademik.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Detectra AI")

    st.markdown(
        """
        Sistem deteksi teks berbasis **BERT** untuk memperkirakan apakah
        sebuah teks lebih menyerupai:

        - Teks Manusia
        - Teks AI
        """
    )

    st.divider()

    st.subheader("Informasi Model")
    st.write("Model: `bert-base-uncased`")
    st.write("Klasifikasi: Biner")
    st.write("Maksimum input: 128 token")
    st.write("Bahasa dataset: Inggris")

    st.divider()

    st.subheader("Petunjuk")
    st.write(
        "Masukkan paragraf berbahasa Inggris, lalu klik tombol "
        "**Deteksi Sekarang**."
    )

    st.warning(
        "Model dilatih menggunakan teks bahasa Inggris. "
        "Hasil untuk teks bahasa Indonesia belum dapat dianggap valid "
        "tanpa evaluasi tambahan."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-badge">BERT TEXT CLASSIFICATION</div>
        <div class="hero-title">Detectra AI</div>
        <div class="hero-description">
            Sistem klasifikasi teks untuk memperkirakan apakah sebuah tulisan
            memiliki pola yang lebih dekat dengan konten manusia atau konten
            yang dihasilkan oleh kecerdasan buatan.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT AREA
# ============================================================

input_col, info_col = st.columns([1.65, 1], gap="large")

with input_col:
    st.subheader("Masukkan Teks")

    user_text = st.text_area(
        "Teks yang akan dianalisis",
        placeholder=(
            "Paste an English paragraph here. Example: "
            "Artificial intelligence has changed the way students "
            "access information and complete academic tasks..."
        ),
        height=260
    )

    word_count = count_words(user_text)
    character_count = len(user_text)

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.metric("Jumlah Kata", word_count)

    with stat_col2:
        st.metric("Jumlah Karakter", character_count)

    detect_button = st.button(
        "Deteksi Sekarang",
        type="primary",
        use_container_width=True
    )


with info_col:
    st.subheader("Cara Kerja Sistem")

    st.markdown(
        """
        <div class="step-card">
            <span class="step-number">1</span>
            <span class="step-title">Input Teks</span>
            <div class="step-description">
                Pengguna memasukkan paragraf yang akan dianalisis.
            </div>
        </div>

        <div class="step-card">
            <span class="step-number">2</span>
            <span class="step-title">Tokenisasi</span>
            <div class="step-description">
                Teks diubah menjadi token input yang dapat dipahami BERT.
            </div>
        </div>

        <div class="step-card">
            <span class="step-number">3</span>
            <span class="step-title">Prediksi Model</span>
            <div class="step-description">
                Model menghitung probabilitas teks manusia dan teks AI.
            </div>
        </div>

        <div class="step-card">
            <span class="step-number">4</span>
            <span class="step-title">Hasil Analisis</span>
            <div class="step-description">
                Sistem menampilkan kelas prediksi dan keyakinan model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROSES DETEKSI
# ============================================================

if detect_button:
    cleaned_input = normalize_text(user_text)

    if not cleaned_input:
        st.warning("Masukkan teks terlebih dahulu.")

    elif count_words(cleaned_input) < 10:
        st.warning(
            "Teks terlalu pendek. Masukkan minimal 10 kata "
            "agar hasil prediksi lebih representatif."
        )

    else:
        with st.spinner("Menganalisis teks..."):
            start_time = time.time()

            result = predict_text(cleaned_input)

            processing_time = time.time() - start_time

        st.divider()
        st.subheader("Hasil Deteksi")

        display_result(result)

        st.caption(
            f"Waktu pemrosesan: {processing_time:.3f} detik | "
            f"Model: {MODEL_REPO}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Detectra AI — Deteksi Teks AI dan Manusia Menggunakan Model BERT
    </div>
    """,
    unsafe_allow_html=True
)
