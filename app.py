import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agro Bromo Sentiment Analysis",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{

    --primary:#FF7A00;
    --primary-soft:#FFA64D;

    --bg:#FFFDFB;
    --surface:#FFFFFF;

    --text:#1E1E1E;
    --muted:#737373;

    --border:#EFEFEF;

    --positive:#16A34A;
    --neutral:#D97706;
    --negative:#DC2626;

}

/* ================================================= */
/* BASE */
/* ================================================= */

html, body, .stApp{
    background:var(--bg);
    color:var(--text);
    font-family:'Plus Jakarta Sans', sans-serif;
}

/* ================================================= */
/* SIDEBAR */
/* ================================================= */

section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #FFF6EC 0%,
        #FFFFFF 100%
    ) !important;

    border-right:1px solid var(--border);

    min-width:280px !important;
    max-width:280px !important;
}

section[data-testid="stSidebar"] *{
    color:#1E1E1E !important;
    font-family:'Plus Jakarta Sans', sans-serif !important;
}

/* ================================================= */
/* HIDE */
/* ================================================= */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

/* ================================================= */
/* HERO */
/* ================================================= */

.hero{
    padding:42px;
    border-radius:30px;

    background:linear-gradient(
        135deg,
        #FF7A00 0%,
        #FFA64D 100%
    );

    color:white;

    margin-bottom:28px;

    box-shadow:0 14px 30px rgba(255,122,0,0.18);
}

.hero-title{
    font-size:52px;
    font-weight:800;
    line-height:1.1;
}

.hero-sub{
    margin-top:12px;
    font-size:16px;
    line-height:1.8;
    opacity:0.95;
}

/* ================================================= */
/* CARD */
/* ================================================= */

.card{
    background:white;

    border:1px solid var(--border);

    padding:24px;

    border-radius:24px;

    box-shadow:0 5px 16px rgba(0,0,0,0.04);

    transition:0.25s;
}

.card:hover{
    transform:translateY(-4px);
    box-shadow:0 10px 28px rgba(0,0,0,0.08);
}

/* ================================================= */
/* METRIC */
/* ================================================= */

.metric-number{
    font-size:42px;
    font-weight:800;
    color:var(--primary);
}

.metric-label{
    font-size:13px;
    color:var(--muted);
    margin-top:4px;
}

/* ================================================= */
/* SECTION */
/* ================================================= */

.section-title{
    font-size:30px;
    font-weight:800;
    margin-bottom:18px;
    margin-top:8px;
}

/* ================================================= */
/* BUTTON */
/* ================================================= */

.stButton > button{

    background:linear-gradient(
        135deg,
        #FF7A00 0%,
        #FFA64D 100%
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:14px !important;

    padding:12px 18px !important;

    font-weight:700 !important;

    transition:0.25s !important;
}

.stButton > button:hover{
    transform:translateY(-2px);
    box-shadow:0 10px 20px rgba(255,122,0,0.22);
}

/* ================================================= */
/* INPUT */
/* ================================================= */

.stTextArea textarea{
    border-radius:18px !important;
    border:1px solid #ECECEC !important;
    padding:16px !important;
}

.stTextInput input{
    border-radius:14px !important;
}

.stTextArea textarea:focus{
    border:1px solid #FF7A00 !important;
    box-shadow:0 0 0 2px rgba(255,122,0,0.12) !important;
}

/* ================================================= */
/* TABS */
/* ================================================= */

button[data-baseweb="tab"]{
    font-weight:700 !important;
    color:#777 !important;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#FF7A00 !important;
    border-bottom:3px solid #FF7A00 !important;
}

/* ================================================= */
/* DATAFRAME */
/* ================================================= */

[data-testid="stDataFrame"]{
    border-radius:20px;
    overflow:hidden;
    border:1px solid #F0F0F0;
}

/* ================================================= */
/* BADGE */
/* ================================================= */

.badge{
    padding:6px 16px;
    border-radius:999px;
    font-size:12px;
    font-weight:700;
}

.badge-positive{
    background:#DCFCE7;
    color:#15803D;
}

.badge-neutral{
    background:#FEF3C7;
    color:#B45309;
}

.badge-negative{
    background:#FEE2E2;
    color:#B91C1C;
}

/* ================================================= */
/* PROGRESS */
/* ================================================= */

.stProgress > div > div > div{
    background:#FF7A00 !important;
}

/* ================================================= */
/* FILE UPLOADER */
/* ================================================= */

[data-testid="stFileUploader"]{
    border-radius:18px;
    border:2px dashed #FFD0A1;
    background:#FFF9F3;
}

/* ================================================= */
/* INFO BOX */
/* ================================================= */

.info-box{
    background:#FFF4E7;
    border:1px solid #FFD7AA;
    padding:18px;
    border-radius:18px;
    color:#9A5C00;
    margin-bottom:22px;
}

/* ================================================= */
/* HR */
/* ================================================= */

hr{
    border:none;
    border-top:1px solid #F1F1F1;
    margin:20px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# NLP
# =========================================================

@st.cache_resource
def load_nlp():

    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')

    from nltk.corpus import stopwords

    return stopwords.words('indonesian')

# =========================================================
# CLEANING
# =========================================================

SLANG = {

    "gk":"tidak",
    "ga":"tidak",
    "gak":"tidak",
    "nggak":"tidak",
    "bgt":"banget",
    "bangettt":"banget",
    "klo":"kalau",
    "krn":"karena",
    "yg":"yang",
    "dr":"dari",
    "aja":"saja",
    "udh":"sudah",
    "org":"orang",
    "tp":"tapi",
    "jd":"jadi",
    "gw":"saya",
    "gue":"saya"

}

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize(text):

    words = text.split()

    normalized = [SLANG.get(w, w) for w in words]

    return " ".join(normalized)

def preprocess(text):

    text = clean_text(text)

    text = normalize(text)

    return text

# =========================================================
# DUMMY SENTIMENT
# =========================================================

def predict_sentiment(text):

    text = preprocess(text)

    negative_words = [
        "buruk",
        "parah",
        "sedih",
        "gagal",
        "marah",
        "jelek",
        "lambat"
    ]

    positive_words = [
        "bagus",
        "baik",
        "hebat",
        "cepat",
        "mantap",
        "terima kasih"
    ]

    neg = sum([1 for w in negative_words if w in text])

    pos = sum([1 for w in positive_words if w in text])

    if neg > pos:

        return "Negative", np.random.uniform(0.85,0.99)

    elif pos > neg:

        return "Positive", np.random.uniform(0.85,0.99)

    else:

        return "Neutral", np.random.uniform(0.70,0.90)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="
        padding-top:10px;
        padding-bottom:20px;
    ">

    <div style="
        font-size:32px;
        font-weight:800;
        color:#FF7A00;
    ">
        🚄 Agro Bromo
    </div>

    <div style="
        color:#666;
        font-size:13px;
        line-height:1.7;
        margin-top:8px;
    ">
        Web Mining & Sentiment Analysis Dashboard
    </div>

    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(

        "Navigation",

        [

            "🏠 Dashboard",
            "🧠 Analisis Sentimen",
            "📊 Statistik & EDA",
            "☁️ WordCloud",
            "📁 Batch Prediction",
            "📈 Machine Learning",
            "📌 Tentang"

        ]

    )

    st.markdown("---")

    model_choice = st.selectbox(

        "Model",

        [

            "RoBERTa",
            "IndoBERT"

        ]

    )

    preprocessing = st.toggle(

        "Show Preprocessing",
        value=False

    )

# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            Agro Bromo Sentiment Analysis 🚄
        </div>

        <div class="hero-sub">
            Dashboard analisis sentimen komentar YouTube terhadap berita kecelakaan Kereta Api Argo Bromo menggunakan NLP, IndoBERT, RoBERTa, Random Forest, dan XGBoost.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # METRICS
    # =====================================================

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.markdown("""
        <div class="card">

            <div class="metric-number">
                3
            </div>

            <div class="metric-label">
                Channel YouTube
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

            <div class="metric-number">
                2
            </div>

            <div class="metric-label">
                Transformer Models
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

            <div class="metric-number">
                RF
            </div>

            <div class="metric-label">
                Random Forest
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c4:

        st.markdown("""
        <div class="card">

            <div class="metric-number">
                XGB
            </div>

            <div class="metric-label">
                XGBoost
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PIPELINE
    # =====================================================

    st.markdown(

        '<div class="section-title">📌 Pipeline Penelitian</div>',
        unsafe_allow_html=True

    )

    p1,p2,p3,p4,p5,p6 = st.columns(6)

    pipeline = [

        ("📥","Web Crawling"),
        ("🧹","Preprocessing"),
        ("🧠","Transformer"),
        ("📊","EDA"),
        ("🤖","Machine Learning"),
        ("📈","Insight")

    ]

    cols = [p1,p2,p3,p4,p5,p6]

    for col, item in zip(cols, pipeline):

        with col:

            st.markdown(f"""
            <div class="card" style="text-align:center;">

                <div style="font-size:42px;">
                    {item[0]}
                </div>

                <div style="
                    margin-top:10px;
                    font-weight:700;
                ">
                    {item[1]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CHANNELS
    # =====================================================

    st.markdown(

        '<div class="section-title">📺 Sumber Dataset</div>',
        unsafe_allow_html=True

    )

    a,b,c = st.columns(3)

    channels = [

        ("🔴 TVONE"),
        ("🔵 KOMPAS TV"),
        ("🟢 METRO TV")

    ]

    for col, name in zip([a,b,c], channels):

        with col:

            st.markdown(f"""
            <div class="card">

                <div style="
                    font-size:22px;
                    font-weight:800;
                ">
                    {name}
                </div>

                <div style="
                    color:#666;
                    line-height:1.8;
                    margin-top:12px;
                ">
                    Data komentar YouTube digunakan sebagai sumber utama untuk analisis sentimen publik.
                </div>

            </div>
            """, unsafe_allow_html=True)

# =========================================================
# ANALISIS SENTIMEN
# =========================================================

elif menu == "🧠 Analisis Sentimen":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            Analisis Sentimen
        </div>

        <div class="hero-sub">
            Analisis komentar YouTube secara real-time menggunakan NLP.
        </div>

    </div>
    """, unsafe_allow_html=True)

    ex1,ex2,ex3 = st.columns(3)

    with ex1:

        if st.button("😔 Contoh Negatif"):

            st.session_state.text = "pelayanan sangat buruk dan lambat"

    with ex2:

        if st.button("😐 Contoh Netral"):

            st.session_state.text = "kereta mengalami gangguan teknis"

    with ex3:

        if st.button("😊 Contoh Positif"):

            st.session_state.text = "terima kasih tim penyelamat"

    text = st.text_area(

        "Input Text",

        value=st.session_state.get("text",""),

        height=180,

        placeholder="Masukkan komentar YouTube di sini..."

    )

    if preprocessing and text:

        st.markdown("### 🔧 Hasil Preprocessing")

        st.code(preprocess(text))

    if st.button("🚀 Jalankan Analisis", use_container_width=True):

        if text.strip() == "":

            st.warning("Masukkan teks terlebih dahulu.")

        else:

            sentiment, score = predict_sentiment(text)

            if sentiment == "Positive":

                badge = "badge-positive"
                emoji = "😊"

            elif sentiment == "Neutral":

                badge = "badge-neutral"
                emoji = "😐"

            else:

                badge = "badge-negative"
                emoji = "😔"

            st.markdown(f"""
            <div class="card">

                <div style="
                    font-size:24px;
                    font-weight:800;
                    margin-bottom:20px;
                ">
                    Hasil Prediksi
                </div>

                <div style="
                    text-align:center;
                    font-size:82px;
                ">
                    {emoji}
                </div>

                <div style="text-align:center;">
                    <span class="badge {badge}">
                        {sentiment}
                    </span>
                </div>

                <div style="
                    text-align:center;
                    margin-top:18px;
                    color:#666;
                ">
                    Confidence Score
                </div>

                <div style="
                    text-align:center;
                    font-size:44px;
                    font-weight:800;
                    color:#FF7A00;
                ">
                    {score*100:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)
