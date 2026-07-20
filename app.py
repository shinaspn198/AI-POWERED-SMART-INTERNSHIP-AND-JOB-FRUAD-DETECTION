import os
import re
import pickle
import streamlit as st
import streamlit.components.v1 as components
from scipy.sparse import hstack, csr_matrix
from lime.lime_text import LimeTextExplainer

try:
    from rag_engine import rag_answer
except Exception as e:
    _rag_error = str(e)
    def rag_answer(question, document_text=None):
        return f"RAG engine could not load: {_rag_error}"

try:
    import pdfplumber
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    import shutil

    # Automatically find Tesseract
    tesseract = shutil.which("tesseract")

    if tesseract:
        pytesseract.pytesseract.tesseract_cmd = tesseract
        OCR_AVAILABLE = True
    else:
        OCR_AVAILABLE = False

except Exception:
    OCR_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Smart Job Fraud Detection AI",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
html, body, [class*="css"] { color: #ffffff; }
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 25%),
        radial-gradient(circle at bottom right, rgba(6,182,212,0.16), transparent 25%),
        linear-gradient(135deg, #020617 0%, #071225 45%, #0f172a 100%);
    background-attachment: fixed;
}
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }
.hero {
    position: relative; overflow: hidden;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 34px; padding: 50px; margin-bottom: 28px;
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.45), 0 0 60px rgba(37,99,235,0.10);
}
.hero::before {
    content: ""; position: absolute;
    width: 420px; height: 420px; top: -180px; right: -150px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(37,99,235,0.28), transparent 70%);
}
.hero::after {
    content: ""; position: absolute;
    width: 320px; height: 320px; bottom: -180px; left: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.18), transparent 70%);
}
.hero-title {
    position: relative; z-index: 2;
    font-size: 56px; line-height: 1.05; font-weight: 900; color: white;
    letter-spacing: -2px; margin-bottom: 18px;
}
.hero-title span {
    background: linear-gradient(135deg, #60a5fa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    position: relative; z-index: 2; color: #cbd5e1; font-size: 17px;
    line-height: 1.8; max-width: 900px; margin-bottom: 28px;
}
.hero-pills { position: relative; z-index: 2; display: flex; flex-wrap: wrap; gap: 12px; }
.hero-pill {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(96,165,250,0.22);
    color: #dbeafe; padding: 11px 16px; border-radius: 999px;
    font-size: 13px; font-weight: 800; transition: 0.3s ease;
}
.hero-pill:hover { transform: translateY(-3px); background: rgba(37,99,235,0.18); }
.panel {
    background: rgba(15,23,42,0.74); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 30px; padding: 30px; margin-bottom: 24px;
    backdrop-filter: blur(16px); box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}
.section-title { color: white; font-size: 30px; font-weight: 900; margin-bottom: 10px; }
.section-desc  { color: #94a3b8; font-size: 15px; margin-bottom: 22px; }
.upload-note {
    background: rgba(37,99,235,0.12); border: 1px solid rgba(96,165,250,0.20);
    color: #dbeafe; border-radius: 16px; padding: 14px 16px;
    margin-bottom: 20px; font-weight: 700;
}
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.65) !important;
    border: 2px dashed rgba(96,165,250,0.35) !important;
    border-radius: 26px !important; padding: 20px !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p { color: #cbd5e1 !important; }
textarea, input {
    background: rgba(15,23,42,0.88) !important; color: white !important;
    border: 1px solid rgba(96,165,250,0.20) !important;
    border-radius: 18px !important; font-weight: 600 !important;
}
textarea:focus, input:focus {
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
    color: white !important; border: none !important;
    border-radius: 18px !important; padding: 0.9rem 1.6rem !important;
    font-weight: 800 !important; transition: 0.3s ease !important;
    box-shadow: 0 10px 35px rgba(37,99,235,0.30) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 18px 45px rgba(37,99,235,0.45) !important;
}
button[data-baseweb="tab"] {
    background: rgba(15,23,42,0.85) !important; color: #e2e8f0 !important;
    border-radius: 18px 18px 0 0 !important; padding: 14px 24px !important;
    font-weight: 800 !important; border: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #06b6d4) !important; color: white !important;
}
.result-fraud, .result-legit, .result-suspicious {
    border-radius: 28px; padding: 30px; margin-bottom: 24px; backdrop-filter: blur(14px);
}
.result-fraud      { background: rgba(127,29,29,0.20); border: 1px solid rgba(248,113,113,0.22); }
.result-legit      { background: rgba(20,83,45,0.20);  border: 1px solid rgba(74,222,128,0.22); }
.result-suspicious { background: rgba(120,53,15,0.20); border: 1px solid rgba(251,191,36,0.22); }
.result-title { font-size: 34px; font-weight: 900; color: white; }
.result-desc  { color: #cbd5e1; margin-top: 8px; }
.score-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; margin: 28px 0; }
.score-card {
    background: rgba(15,23,42,0.82); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px; padding: 28px; text-align: center; transition: 0.3s ease;
}
.score-card:hover { transform: translateY(-6px); }
.score-label { color: #94a3b8; font-size: 13px; font-weight: 800; text-transform: uppercase; }
.score-value { color: white; font-size: 36px; font-weight: 900; margin-top: 10px; }
.xai-card {
    background: rgba(15,23,42,0.75); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 12px 16px; margin-bottom: 8px;
    color: #e2e8f0; font-size: 14px;
}
.fraud-chip {
    display: inline-block; background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3); color: #fca5a5;
    border-radius: 999px; padding: 5px 12px; margin: 4px;
    font-size: 12px; font-weight: 700;
}
.legit-chip {
    display: inline-block; background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3); color: #86efac;
    border-radius: 999px; padding: 5px 12px; margin: 4px;
    font-size: 12px; font-weight: 700;
}
.chat-answer {
    background: rgba(15,23,42,0.85); border: 1px solid rgba(96,165,250,0.18);
    border-left: 5px solid #3b82f6; border-radius: 22px;
    padding: 22px; margin-top: 18px; color: white;
}
@media(max-width:900px) {
    .hero { padding: 35px; }
    .hero-title { font-size: 38px; }
    .score-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    m  = pickle.load(open(os.path.join(BASE_DIR, "model/fraud_model.pkl"), "rb"))
    v  = pickle.load(open(os.path.join(BASE_DIR, "model/vectorizer.pkl"), "rb"))
    fk = pickle.load(open(os.path.join(BASE_DIR, "model/fraud_keywords.pkl"), "rb"))
    try:
        gk = pickle.load(open(os.path.join(BASE_DIR, "model/genuine_keywords.pkl"), "rb"))
    except Exception:
        gk = [
            "official company", "interview scheduled", "hr department",
            "technical round", "offer letter", "appointment letter",
            "background verification", "joining date", "annual ctc",
            "permanent position", "employment contract", "registered company",
            "corporate email", "linkedin profile", "official website",
            "salary package", "provident fund", "medical insurance",
            "department", "joining date"
        ]
        pickle.dump(gk, open(os.path.join(BASE_DIR, "model/genuine_keywords.pkl"), "wb"))
    return m, v, fk, gk

@st.cache_resource
def load_explainer():
    return LimeTextExplainer(class_names=["Legit", "Fraud"])

model, vectorizer, FRAUD_KEYWORDS, GENUINE_KEYWORDS = load_model()
explainer = load_explainer()

FRAUD_REGEX_PATTERNS = {
    "consultant deduction": r"deduct(ed)?\s+from\s+(first\s+)?month|taken\s+by\s+your\s+consultant",
    "sign and return urgency": r"sign\s+(and\s+)?return|return\s+scan\s+copy|within\s+\d+\s+hour|signing\s+the\s+written\s+contract\s+and\s+return|sign.*return",
    "fake email domain": r"@\w+official\.com|@\w+uae\.com|@\w+jobs\.com|\w+official\.com|\w+uae\.com|\w+jobs\.com",
    "soft copy offer": r"soft\s+copy\s+of\s+official\s+letter|soft\s+copy\s+appointment|soft\s+copy\s+of\s+official\s+letter\s+of\s+appointment",
    "too many benefits": r"free\s+accommodation.*food.*medical.*transport|accommodation.*food.*medical.*transport",
    "direct appointment": r"offer\s+of\s+employment|letter\s+of\s+appointment|appointment\s+letter",
}

def fix_ocr_text(text):
    fixes = {
        "registration changes": "registration charges",
        "direct oppose": "direct deposit fraud",
        "tmist": "trust",
        "cxempted": "exempted",
        "bby ": "by ",
        "10 offer": "to offer",
        "pleascd": "pleased",
        "ofTer": "offer",
        "datc": "date",
        "pleasc": "please",
        "rcceive": "receive",
        "submlt": "submit",
        "Iimited": "limited",
        "llmited": "limited",
        "dpworldofficial com": "dpworldofficial.com",
    }
    for wrong, correct in fixes.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s%₹$@.]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def fraud_kw_count(text):
    text_lower = str(text).lower()

    keyword_count = sum(
        1 for kw in FRAUD_KEYWORDS
        if str(kw).lower() in text_lower
    )

    regex_count = sum(
        1 for pattern in FRAUD_REGEX_PATTERNS.values()
        if re.search(pattern, text_lower)
    )

    return keyword_count + regex_count

def genuine_kw_count(text):
    return sum(1 for kw in GENUINE_KEYWORDS if kw in str(text).lower())

def caps(text):
    return sum(1 for w in str(text).split() if w.isupper())

def ex(text):
    return str(text).count("!")

def length(text):
    return len(str(text).split())

def wa(text):
    return int("whatsapp" in str(text).lower())

def sal(text):
    return int(bool(re.search(r"(earn|salary|stipend)\s*[\$₹]?\s*\d{4,}", str(text).lower())))

def payment_pattern(text):
    return int(bool(re.search(
        r"(pay|payment|fee|deposit|charges|changes|transfer|deduct)\s*(₹|\$|inr|usd|aed)?\s*\d+"
        r"|deducted from (first )?month"
        r"|taken by your consultant",
        str(text).lower()
    )))

def interview_pattern(text):
    return int(bool(re.search(r"(technical interview|hr round|interview scheduled|selected after interview)", str(text).lower())))

def transform_input(texts):
    cleaned = [clean_text(t) for t in texts]
    tfidf   = vectorizer.transform(cleaned)
    extra   = []
    for raw, cl in zip(texts, cleaned):
        extra.append([
            fraud_kw_count(cl), genuine_kw_count(cl),
            caps(raw), ex(raw), length(cl),
            wa(cl), sal(cl), payment_pattern(cl), interview_pattern(cl)
        ])
    return hstack([tfidf, csr_matrix(extra)])

def get_strong_fraud_reasons(text):
    text_lower = str(text).lower()
    reasons = [kw for kw in FRAUD_KEYWORDS if str(kw).lower() in text_lower]

    patterns = {
        "payment request": r"(pay|payment|fee|deposit|amount|transfer|charges|changes|deduct)\s*(₹|\$|inr|usd|aed)?\s*\d+",
        "no interview claim": r"no\s+interview",
        "direct selection": r"direct\s+selection|direct\s+joining",
        "whatsapp-only contact": r"whatsapp\s+only|contact\s+on\s+whatsapp|send\s+resume\s+on\s+whatsapp",
        "guaranteed job/salary": r"guaranteed\s+(job|salary|income)",
        "urgent pressure": r"urgent|immediately|limited\s+seats|apply\s+immediately|72\s+hours|24\s+hours",
        "certificate payment": r"(certificate|certification).*(fee|payment|pay)|pay.*(certificate|certification)",
        "registration payment": r"submit.*registration|registration.*(charges|changes|fee)",
    }

    for reason, pattern in patterns.items():
        if re.search(pattern, text_lower):
            reasons.append(reason)

    for reason, pattern in FRAUD_REGEX_PATTERNS.items():
        if re.search(pattern, text_lower):
            reasons.append(reason)

    return list(dict.fromkeys(reasons))

def predict(text):
    X     = transform_input([text])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    strong_reasons = get_strong_fraud_reasons(text)

    high_risk_rules = [
        "consultant deduction",
        "sign and return urgency",
        "fake email domain",
        "soft copy offer",
        "too many benefits",
        "payment request",
        "registration payment",
        "certificate payment",
        "no interview claim",
        "direct selection",
        "whatsapp-only contact"
    ]

    matched_high_risk = [r for r in strong_reasons if r in high_risk_rules]

    if len(matched_high_risk) >= 2:
        pred = 1
        proba[1] = max(proba[1], 0.85)
        proba[0] = 1 - proba[1]

    label = "Fraud" if pred == 1 else "Legit"
    return label, proba[pred] * 100, proba

def predict_proba_fn(texts):
    return model.predict_proba(transform_input(texts))

def get_risk_level(fraud_score):
    if fraud_score >= 65:   return "Fraud",      "🔴"
    elif fraud_score >= 35: return "Suspicious",  "🟡"
    else:                   return "Legit",       "🟢"

def is_generic_word(word):
    clean = re.sub(r"[^a-zA-Z0-9]", "", str(word).strip().lower())

    ignore_words = [
        "internship", "job", "offer", "company", "certificate",
        "contact", "date", "name", "email", "phone",
        "your", "you", "we", "our", "the", "and", "with", "shall",
        "dubai", "duba", "uae", "india", "kerala", "calicut"
    ]

    return len(clean) <= 2 or clean in ignore_words

def show_human_xai(label, explanation, final_text):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Explainable AI Reasoning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Fraud indicators, genuine signals, and model word-impact explanation.</div>', unsafe_allow_html=True)

    fraud_reasons = get_strong_fraud_reasons(final_text)

    if label in ["Fraud", "Suspicious"]:
        if fraud_reasons:
            st.error("Suspicious risk indicators: " + ", ".join(fraud_reasons[:7]))
            chips = "".join([f"<span class='fraud-chip'>⚠️ {r}</span>" for r in fraud_reasons[:9]])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.error("This document looks suspicious based on wording and structure.")
    else:
        genuine_reasons = [kw for kw in GENUINE_KEYWORDS if kw.lower() in final_text.lower()]
        if genuine_reasons:
            st.success("Professional indicators: " + ", ".join(genuine_reasons[:7]))
            chips = "".join([f"<span class='legit-chip'>✅ {r}</span>" for r in genuine_reasons[:9]])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("This document contains professional and legitimate indicators.")

    st.markdown("#### 🔍 Word Impact (LIME)")
    shown = False
    for word, score in explanation:
        if is_generic_word(word):
            continue
        shown = True
        if score > 0:
            st.markdown(f"<div class='xai-card'>⚠️ <b>{word}</b> — increased fraud risk</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='xai-card'>✅ <b>{word}</b> — supported legitimacy</div>", unsafe_allow_html=True)

    if not shown:
        st.markdown("<div class='xai-card'>ℹ️ Prediction based on document structure, keyword signals, and contextual indicators.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def show_lime_clean(exp):
    html = exp.as_html()
    clean_html = f"""
    <html><head><style>
    body {{ background:#ffffff !important; color:#111827 !important;
           font-family:Arial,sans-serif !important; padding:18px !important; }}
    div,span,p,h1,h2,h3,h4,h5,h6,text {{ color:#111827 !important; }}
    .lime {{ background:#ffffff !important; color:#111827 !important; }}
    svg text {{ fill:#111827 !important; }}
    table,td,th {{ color:#111827 !important; }}
    </style></head><body>{html}</body></html>"""
    components.html(clean_html, height=620, scrolling=True)

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file is None:
        return text

    file_type = uploaded_file.type

    if file_type == "text/plain":
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif "pdf" in file_type:
        if PDF_AVAILABLE:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        else:
            st.error("pdfplumber not installed.")

    elif "image" in file_type:
        if OCR_AVAILABLE:
            image = Image.open(uploaded_file)
            raw   = pytesseract.image_to_string(image)
            text  = fix_ocr_text(raw)
        else:
            st.error("OCR not available.")

    return text

st.markdown("""
<div class="hero">
    <div class="hero-title">
        Detect. Explain. <span>Verify.</span><br>
        Stay Safe from <span>Job Scams.</span>
    </div>
    <div class="hero-subtitle">
        Our AI analyzes job offers, internship certificates, and recruitment documents
        to detect fraud, explain the reasoning, and answer your questions using RAG.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">🤖 Machine Learning</span>
        <span class="hero-pill">🧾 OCR &amp; PDF</span>
        <span class="hero-pill">🧠 LIME XAI</span>
        <span class="hero-pill">💬 RAG Q&amp;A</span>
        <span class="hero-pill">🚨 Fraud Risk Scoring</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📄 Analyze Document", "💬 AI Document Assistant"])

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

with tab1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload or Paste Document</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Upload an offer letter, internship certificate, job PDF, screenshot, or paste text manually.</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-note">📎 Supported: PDF, PNG, JPG, JPEG, TXT</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg", "txt"])
    with col2:
        manual_text = st.text_area(
            "Paste Document Text", height=230,
            placeholder="Paste job offer, internship message, email, or certificate text..."
        )

    extracted_text = ""
    if uploaded_file is not None:
        extracted_text = extract_text_from_file(uploaded_file)

    final_text = manual_text.strip() if manual_text.strip() else extracted_text.strip()

    if final_text:
        st.session_state.document_text = final_text
        with st.expander("📄 Text Preview"):
            st.write(final_text[:3000])

    analyze = st.button("🔍 Analyze Fraud Risk")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not final_text:
            st.warning("Please upload a document or paste text first.")
        else:
            with st.spinner("🤖 Analyzing..."):
                label, confidence, proba = predict(final_text)
                fraud_score = proba[1] * 100
                legit_score = proba[0] * 100
                risk_level, risk_icon = get_risk_level(fraud_score)

            if risk_level == "Fraud":
                st.markdown("""<div class="result-fraud">
                    <div class="result-title">🚨 Fraud Detected</div>
                    <div class="result-desc">Suspicious indicators found. Do NOT pay any fees or share personal details.</div>
                </div>""", unsafe_allow_html=True)
            elif risk_level == "Suspicious":
                st.markdown("""<div class="result-suspicious">
                    <div class="result-title">⚠️ Suspicious Document</div>
                    <div class="result-desc">Some red flags detected. Verify the company carefully before proceeding.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-legit">
                    <div class="result-title">✅ Genuine / Legit Document</div>
                    <div class="result-desc">This document looks professional and does not strongly match fraud patterns.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="score-grid">
                <div class="score-card">
                    <div class="score-label">🎯 Confidence</div>
                    <div class="score-value">{confidence:.1f}%</div>
                </div>
                <div class="score-card">
                    <div class="score-label">🚨 Fraud Probability</div>
                    <div class="score-value">{fraud_score:.1f}%</div>
                </div>
                <div class="score-card">
                    <div class="score-label">✅ Legit Probability</div>
                    <div class="score-value">{legit_score:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.progress(int(fraud_score))

            with st.spinner("🧠 Generating LIME explanation..."):
                exp = explainer.explain_instance(
                    final_text, predict_proba_fn,
                    num_features=12,
                    num_samples=500
                )

            show_human_xai(label, exp.as_list(), final_text)

            with st.expander("📊 Full LIME Visual"):
                show_lime_clean(exp)

with tab2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 AI Document Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Ask questions about the uploaded document — company name, certificate ID, duration, salary, signer, or why it looks fraudulent.</div>', unsafe_allow_html=True)

    if not st.session_state.document_text:
        st.info("📎 Upload or paste a document in the Analyze tab first.")
    else:
        st.success(f"✅ Document ready — {len(st.session_state.document_text.split())} words loaded")

        st.markdown("**💡 Quick questions:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Who signed this?"):
                st.session_state.rag_q = "Who signed this document?"
        with col_b:
            if st.button("What is the company?"):
                st.session_state.rag_q = "What is the company name?"
        with col_c:
            if st.button("What is the duration?"):
                st.session_state.rag_q = "What is the internship duration?"

        question = st.text_input(
            "Ask about the document",
            value=st.session_state.get("rag_q", ""),
            placeholder="Who signed the certificate? What is the stipend? Is this legitimate?"
        )

        if st.button("🤖 Ask AI Assistant"):
            if question.strip():
                with st.spinner("🔍 Searching document..."):
                    answer = rag_answer(question, document_text=st.session_state.document_text)
                st.markdown(
                    f"<div class='chat-answer'><b>🤖 AI Assistant</b><br><br>{answer}</div>",
                    unsafe_allow_html=True
                )
                if "rag_q" in st.session_state:
                    del st.session_state["rag_q"]
            else:
                st.warning("Please type a question.")

    st.markdown("</div>", unsafe_allow_html=True)
