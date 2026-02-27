"""
SwinEff-DR  ·  Diabetic Retinopathy Detection System
─────────────────────────────────────────────────────
Streamlit application for retinal fundus image analysis
using the hybrid Swin Transformer & EfficientNet model.
"""

import os, sys, time, io
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="SwinEff-DR · Diabetic Retinopathy Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero gradient background */
.main > div { padding-top: 1rem; }

/* Card styling */
.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.severity-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.sev-0 { background: linear-gradient(135deg, #00b09b, #96c93d); color: #fff; }
.sev-1 { background: linear-gradient(135deg, #4facfe, #00f2fe); color: #1a1a2e; }
.sev-2 { background: linear-gradient(135deg, #f093fb, #f5576c); color: #fff; }
.sev-3 { background: linear-gradient(135deg, #f85032, #e73827); color: #fff; }
.sev-4 { background: linear-gradient(135deg, #8e0000, #1f1c18); color: #fff; }

.metric-number {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.subtitle {
    color: #a0aec0;
    font-size: 0.85rem;
    font-weight: 400;
    margin-bottom: 0.5rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}

/* Upload area */
.stFileUploader > div {
    border: 2px dashed rgba(102, 126, 234, 0.4) !important;
    border-radius: 16px !important;
    background: rgba(102, 126, 234, 0.05) !important;
}

.stFileUploader > div:hover {
    border-color: rgba(102, 126, 234, 0.8) !important;
    background: rgba(102, 126, 234, 0.1) !important;
}

/* Progress bar colours */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Recommendations */
.rec-box {
    background: rgba(102, 126, 234, 0.08);
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}

.rec-box-warn {
    background: rgba(245, 87, 108, 0.08);
    border-left: 4px solid #f5576c;
}

/* Hide default header */
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe NPDR", "PDR"]
CLASS_FULL  = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative DR",
    "Moderate Non-Proliferative DR",
    "Severe Non-Proliferative DR",
    "Proliferative Diabetic Retinopathy",
]
CLASS_COLORS     = ["#00b09b", "#4facfe", "#f093fb", "#f85032", "#8e0000"]
CLASS_BAR_COLORS = ["#96c93d", "#00f2fe", "#f5576c", "#e73827", "#1f1c18"]

RECOMMENDATIONS = {
    0: [
        "✅ Continue regular annual eye examinations",
        "✅ Maintain good blood glucose control (HbA1c < 7%)",
        "✅ Monitor blood pressure and cholesterol levels",
        "✅ Healthy lifestyle with balanced diet and exercise",
    ],
    1: [
        "📋 Schedule follow-up examination in 6–12 months",
        "📋 Optimize diabetes management with your endocrinologist",
        "📋 Regular monitoring of blood glucose recommended",
        "📋 Consider lifestyle modifications if applicable",
    ],
    2: [
        "⚠️ Schedule follow-up in 3–6 months",
        "⚠️ Consult with an ophthalmologist promptly",
        "⚠️ May require treatment intervention",
        "⚠️ Strict glycaemic control is essential",
    ],
    3: [
        "🔴 Urgent ophthalmology referral required",
        "🔴 Follow-up within 2–4 months",
        "🔴 High risk of progression to PDR",
        "🔴 Treatment is likely necessary — discuss options with specialist",
    ],
    4: [
        "🚨 IMMEDIATE ophthalmology consultation required",
        "🚨 Treatment needed to prevent vision loss",
        "🚨 May require laser therapy or anti-VEGF injections",
        "🚨 Close monitoring is essential — do not delay",
    ],
}

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "models", "swineffdr_full.pth")

# ── Model loading (cached) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from model import SwinEffDR
    model = SwinEffDR(num_classes=5)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── Image preprocessing ──────────────────────────────────
def preprocess_image(pil_img, target_size=224):
    """CLAHE-enhanced preprocessing pipeline."""
    img = np.array(pil_img.convert("RGB"))

    # Green channel extraction
    green = img[:, :, 1]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Back to 3-channel
    enhanced_rgb = cv2.merge([enhanced, enhanced, enhanced])

    # Resize
    resized = cv2.resize(enhanced_rgb, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)

    # To tensor + normalise (ImageNet stats)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0), enhanced_rgb


# ── Inference ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, tensor):
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().numpy()
    pred   = int(np.argmax(probs))
    return pred, probs


# ── Plotly probability chart ──────────────────────────────
def make_prob_chart(probs, pred):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs * 100,
        y=CLASS_NAMES,
        orientation='h',
        marker=dict(
            color=[CLASS_BAR_COLORS[i] if i == pred else 'rgba(150,150,170,0.3)'
                   for i in range(5)],
            line=dict(width=0),
        ),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(size=13, color='white', family='Inter'),
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, max(probs) * 100 * 1.25],
            showgrid=False, zeroline=False,
            tickfont=dict(color='#a0aec0', size=11),
            title=dict(text='Confidence (%)', font=dict(color='#a0aec0', size=12)),
        ),
        yaxis=dict(
            tickfont=dict(color='#e2e8f0', size=13, family='Inter'),
            autorange='reversed',
        ),
        margin=dict(l=10, r=40, t=10, b=40),
        height=240,
        font=dict(family='Inter'),
    )
    return fig


# ═══════════════════ SIDEBAR ══════════════════════════════
with st.sidebar:
    st.markdown("# 🔬 SwinEff-DR")
    st.markdown("### Diabetic Retinopathy Detection")
    st.markdown("---")
    st.markdown("""
    **Model Architecture**
    - 🧠 Swin Transformer-B (global)
    - 🔍 EfficientNet-B4 (local)
    - 🔗 Attention-based fusion
    - 📊 5-class classifier
    """)
    st.markdown("---")
    st.markdown("""
    **Performance Metrics**
    | Metric | Score |
    |--------|-------|
    | Accuracy | 82.24% |
    | QWK: | 0.9081 |
    | Best F1 | 0.6740 |
    | AUC-ROC | 0.8583 |
    """)
    st.markdown("---")
    st.markdown("""
    > ⚠️ **Disclaimer:** This is a research prototype.
    > Results must be validated by qualified ophthalmologists.
    > Not intended for clinical diagnosis.
    """)
    st.markdown("---")
    st.markdown(
        '<p style="color:#718096;font-size:0.75rem;text-align:center">'
        'SwinEff-DR v1.0 · Research Prototype · Feb 2025</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════ MAIN AREA ════════════════════════════

# Hero
st.markdown(
    '<h1 style="text-align:center;background:linear-gradient(135deg,#667eea,#764ba2);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.4rem;'
    'font-weight:800;margin-bottom:0">Diabetic Retinopathy Detection</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center;color:#a0aec0;margin-top:4px;margin-bottom:1.5rem">'
    'Upload a retinal fundus image for automated DR severity classification</p>',
    unsafe_allow_html=True,
)

# Upload
uploaded = st.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"],
    help="Drag-and-drop or click to browse. Supported: JPG, JPEG, PNG.",
)

if uploaded is not None:
    pil_img = Image.open(uploaded)

    # ── Show original ──
    col_orig, col_enh = st.columns(2)
    with col_orig:
        st.markdown("#### 📷 Original Image")
        st.image(pil_img, use_container_width=True)

    # ── Preprocess ──
    with st.spinner("Preprocessing & running inference …"):
        tensor, enhanced = preprocess_image(pil_img)

    with col_enh:
        st.markdown("#### 🔬 CLAHE Enhanced")
        st.image(enhanced, use_container_width=True, clamp=True)

    # ── Load model + predict ──
    with st.spinner("Loading SwinEff-DR model …"):
        model = load_model()

    with st.spinner("Analysing retinal image …"):
        start_t = time.time()
        pred, probs = predict(model, tensor)
        elapsed = time.time() - start_t

    # ── Results ──
    st.markdown("---")
    st.markdown(
        '<h2 style="text-align:center;color:#e2e8f0;margin-bottom:0.5rem">'
        '📊 Analysis Results</h2>',
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns([1.2, 1.5, 1.3])

    with r1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">PREDICTED CLASS</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="severity-badge sev-{pred}">{CLASS_NAMES[pred]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:#cbd5e0;margin-top:10px;font-size:0.95rem">'
            f'{CLASS_FULL[pred]}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="metric-number">{probs[pred]*100:.1f}%</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="subtitle">CONFIDENCE</p>', unsafe_allow_html=True)
        st.progress(float(probs[pred]))
        st.markdown(
            f'<p style="color:#718096;font-size:0.78rem;margin-top:8px">'
            f'Inference: {elapsed*1000:.0f} ms · CPU</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">CLASS PROBABILITIES</p>', unsafe_allow_html=True)
        st.plotly_chart(make_prob_chart(probs, pred), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">CLINICAL RECOMMENDATIONS</p>', unsafe_allow_html=True)
        cls = "rec-box-warn" if pred >= 3 else "rec-box"
        for rec in RECOMMENDATIONS[pred]:
            st.markdown(f'<div class="{cls}">{rec}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Detailed probability table ──
    with st.expander("📋 Detailed Probability Breakdown"):
        for i in range(5):
            bar_pct = probs[i] * 100
            highlight = "★" if i == pred else ""
            st.markdown(
                f"**{CLASS_NAMES[i]}** {highlight}  ·  "
                f"`{bar_pct:.2f}%`"
            )
            st.progress(float(probs[i]))

else:
    # ── Empty state ──
    st.markdown(
        '<div style="text-align:center;padding:3rem;border:2px dashed rgba(102,126,234,0.2);'
        'border-radius:16px;margin:2rem auto;max-width:500px">'
        '<p style="font-size:3rem;margin:0">👁️</p>'
        '<p style="color:#a0aec0;font-size:1.1rem;font-weight:500">Upload a retinal fundus image</p>'
        '<p style="color:#718096;font-size:0.85rem">JPG, JPEG, or PNG · Max 200 MB</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🧠", "Hybrid AI", "Swin Transformer + EfficientNet fusion"),
        ("🎯", "82.24% Accuracy", "State-of-the-art DR detection"),
        ("⚡", "Fast Inference", "Results in seconds on CPU"),
        ("📊", "5-Class Output", "Complete severity grading"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(
                f'<div style="text-align:center;background:rgba(102,126,234,0.06);'
                f'border-radius:12px;padding:1.2rem 0.8rem;border:1px solid rgba(102,126,234,0.12)">'
                f'<p style="font-size:2rem;margin:0">{icon}</p>'
                f'<p style="font-weight:600;color:#e2e8f0;margin:4px 0 2px">{title}</p>'
                f'<p style="color:#a0aec0;font-size:0.8rem;margin:0">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
