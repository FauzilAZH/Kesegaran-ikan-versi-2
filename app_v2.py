import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# ============================================
# KONFIGURASI
# ============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ikan_model_deploy.pt")
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = [
    "Busuk Tidak Segar (5-6 Hari)",
    "Sangat Segar (1-2 Hari)",
    "Segar (3-4 Hari)"
]

CLASS_INFO = {
    "Busuk Tidak Segar (5-6 Hari)": {"color": "#e74c3c", "gradient": "linear-gradient(135deg, #e74c3c, #c0392b)", "status": "TIDAK LAYAK KONSUMSI", "icon": "✕"},
    "Sangat Segar (1-2 Hari)": {"color": "#2ecc71", "gradient": "linear-gradient(135deg, #2ecc71, #27ae60)", "status": "SANGAT LAYAK KONSUMSI", "icon": "✓"},
    "Segar (3-4 Hari)": {"color": "#f39c12", "gradient": "linear-gradient(135deg, #f39c12, #e67e22)", "status": "LAYAK KONSUMSI", "icon": "~"},
}

# ============================================
# FUNGSI LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    all_probs = {CLASS_NAMES[i]: probabilities[0][i].item() * 100 for i in range(NUM_CLASSES)}
    return predicted_class, confidence_pct, all_probs


# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="FreshKan - Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)

# ============================================
# CUSTOM CSS - BLUE & GOLD PREMIUM THEME
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== GLOBAL RESET ===== */
    .stApp {
        background: linear-gradient(160deg, #0a1628 0%, #0d1f3c 30%, #112a4e 60%, #0a1628 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ===== HIDE DEFAULT STREAMLIT ELEMENTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== ANIMATED BACKGROUND PARTICLES ===== */
    .bg-particles {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .bg-particles::before, .bg-particles::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        opacity: 0.06;
        animation: float 20s infinite ease-in-out;
    }
    .bg-particles::before {
        width: 600px; height: 600px;
        background: radial-gradient(circle, #c9a227, transparent 70%);
        top: -200px; right: -200px;
    }
    .bg-particles::after {
        width: 500px; height: 500px;
        background: radial-gradient(circle, #1e90ff, transparent 70%);
        bottom: -150px; left: -150px;
        animation-delay: -10s;
    }
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -40px) scale(1.05); }
        50% { transform: translate(-20px, 20px) scale(0.95); }
        75% { transform: translate(40px, 30px) scale(1.02); }
    }

    /* ===== HERO HEADER ===== */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(201,162,39,0.15), rgba(201,162,39,0.05));
        border: 1px solid rgba(201,162,39,0.3);
        color: #c9a227;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 18px;
        border-radius: 50px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #c9a227 50%, #e8d48b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.5);
        font-weight: 300;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .hero-divider {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #c9a227, transparent);
        margin: 1.2rem auto;
        border-radius: 2px;
    }

    /* ===== GLASS CARD ===== */
    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(201,162,39,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ===== RESULT BOX ===== */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
    }
    .result-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 0.8rem;
    }
    .result-class {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .result-confidence {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-status {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        padding: 6px 16px;
        border-radius: 50px;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* ===== PROBABILITY BAR ===== */
    .prob-container {
        margin: 0.8rem 0;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .prob-name {
        font-size: 0.85rem;
        font-weight: 500;
        color: rgba(255,255,255,0.8);
    }
    .prob-value {
        font-size: 0.85rem;
        font-weight: 700;
        color: #c9a227;
    }
    .prob-track {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
        position: relative;
    }
    .prob-fill::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%) !important;
        border-right: 1px solid rgba(201,162,39,0.15) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.9rem;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #c9a227, #e8d48b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #c9a227;
        margin-bottom: 0.6rem;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(201,162,39,0.05), rgba(30,144,255,0.05));
        border: 2px dashed rgba(201,162,39,0.3);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(201,162,39,0.6);
        background: linear-gradient(135deg, rgba(201,162,39,0.08), rgba(30,144,255,0.08));
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: rgba(255,255,255,0.7) !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #c9a227, #b8941f) !important;
        color: #0a1628 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        box-shadow: 0 4px 20px rgba(201,162,39,0.4) !important;
        transform: translateY(-1px);
    }

    /* ===== INFO BOX ===== */
    .upload-prompt {
        background: linear-gradient(135deg, rgba(30,144,255,0.08), rgba(201,162,39,0.05));
        border: 1px solid rgba(30,144,255,0.2);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
    }
    .upload-prompt-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }

    /* ===== IMAGE CONTAINER ===== */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ===== GENERAL TEXT ===== */
    .stMarkdown p, .stMarkdown li {
        color: rgba(255,255,255,0.75);
    }
    h1, h2, h3, h4 {
        color: rgba(255,255,255,0.9) !important;
    }

    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: #c9a227 !important;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255,255,255,0.25);
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }
    .app-footer a {
        color: #c9a227;
        text-decoration: none;
    }

    /* ===== SECTION TITLE ===== */
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #c9a227;
        margin-bottom: 1rem;
    }
</style>
<div class="bg-particles"></div>
""", unsafe_allow_html=True)


# ============================================
# HERO HEADER
# ============================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Deep Learning Powered</div>
    <div class="hero-title">FreshKan</div>
    <div class="hero-divider"></div>
    <div class="hero-subtitle">
        Sistem klasifikasi kesegaran ikan berbasis citra mata ikan menggunakan arsitektur MobileNetV2
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">FreshKan</div>', unsafe_allow_html=True)
    st.markdown(
        "Sistem cerdas untuk mengklasifikasikan tingkat kesegaran ikan "
        "berdasarkan **citra mata ikan** menggunakan model Deep Learning "
        "MobileNetV2 yang telah di-fine-tuning."
    )

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Kelas Klasifikasi</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Sangat Segar** — Usia 1-2 Hari")
    st.markdown("**Segar** — Usia 3-4 Hari")
    st.markdown("**Busuk / Tidak Segar** — Usia 5-6 Hari")

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Tips Pengambilan Foto</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("- Fokuskan pada bagian **mata ikan**")
    st.markdown("- Gunakan pencahayaan yang cukup")
    st.markdown("- Pastikan gambar tajam & tidak buram")
    st.markdown("- Format: JPG, PNG, atau BMP")

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Teknologi</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("- **Model:** MobileNetV2")
    st.markdown("- **Framework:** PyTorch")
    st.markdown("- **Input:** Citra mata ikan 224×224 px")


# ============================================
# UPLOAD SECTION
# ============================================
st.markdown('<div class="section-title">Upload Gambar</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar mata ikan",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload foto mata ikan dalam format JPG, JPEG, PNG, atau BMP.",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="section-title">Gambar Input</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")

    # Load model & predict
    with st.spinner("Menganalisis gambar..."):
        model, device = load_model()
        image_tensor = preprocess_image(image)
        predicted_class, confidence, all_probs = predict(model, image_tensor, device)

    info = CLASS_INFO[predicted_class]

    with col2:
        st.markdown('<div class="section-title">Hasil Analisis</div>', unsafe_allow_html=True)

        # Determine status badge colors
        if "Busuk" in predicted_class:
            badge_bg = "rgba(231,76,60,0.15)"
            badge_border = "rgba(231,76,60,0.4)"
            badge_text = "#e74c3c"
            bar_gradient = "linear-gradient(90deg, #e74c3c, #c0392b)"
        elif "Sangat" in predicted_class:
            badge_bg = "rgba(46,204,113,0.15)"
            badge_border = "rgba(46,204,113,0.4)"
            badge_text = "#2ecc71"
            bar_gradient = "linear-gradient(90deg, #2ecc71, #27ae60)"
        else:
            badge_bg = "rgba(243,156,18,0.15)"
            badge_border = "rgba(243,156,18,0.4)"
            badge_text = "#f39c12"
            bar_gradient = "linear-gradient(90deg, #f39c12, #e67e22)"

        st.markdown(f"""<div class="result-card" style="border-color: {badge_border};">
    <div class="result-card" style="border:none; padding:0; background:none; backdrop-filter:none;">
        <div class="result-label">Klasifikasi</div>
        <div class="result-class" style="color: {badge_text};">{predicted_class}</div>
        <div class="result-confidence" style="color: #ffffff;">{confidence:.1f}%</div>
        <div class="result-status" style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_text};">
            {info['status']}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    # ===== PROBABILITY SECTION =====
    st.markdown("")
    st.markdown('<div class="section-title">Distribusi Probabilitas</div>', unsafe_allow_html=True)

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    prob_html = ""
    for cls_name, prob in sorted_probs:
        cls_info = CLASS_INFO[cls_name]
        color = cls_info["color"]

        if prob == max(p for _, p in sorted_probs):
            fill_gradient = f"linear-gradient(90deg, {color}, {color}dd)"
        else:
            fill_gradient = f"linear-gradient(90deg, rgba(255,255,255,0.15), rgba(255,255,255,0.08))"

        prob_html += f"""<div class="prob-container">
    <div class="prob-header">
        <span class="prob-name">{cls_name}</span>
        <span class="prob-value" style="color: {color};">{prob:.2f}%</span>
    </div>
    <div class="prob-track">
        <div class="prob-fill" style="width: {prob}%; background: {fill_gradient};"></div>
    </div>
</div>
"""

    st.markdown(f"""<div class="glass-card">
{prob_html}
</div>""", unsafe_allow_html=True)

else:
    st.markdown("""<div class="upload-prompt">
    <div class="upload-prompt-icon">&#9650;</div>
    <div>Silakan unggah gambar mata ikan di atas untuk memulai analisis kesegaran.</div>
</div>""", unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("""<div class="app-footer">
    FreshKan &mdash; Sistem Klasifikasi Kesegaran Ikan &bull; Powered by MobileNetV2 &amp; PyTorch
</div>""", unsafe_allow_html=True)
