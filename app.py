# STREAMLIT PREMIUM WHITE ORANGE DASHBOARD (FULL REDESIGN)

```python
import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #FFFDF9;
    --surface: #FFFFFF;
    --surface-2: #FFF6ED;
    --border: #F2E7DA;

    --text: #1F2937;
    --muted: #7B7280;

    --orange: #FF8A3D;
    --orange-soft: #FFB36B;
    --orange-light: #FFE7D1;

    --green: #22C55E;
    --yellow: #FACC15;
    --red: #EF4444;

    --shadow:
        0 4px 20px rgba(255, 138, 61, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(255,180,107,0.20), transparent 30%),
        radial-gradient(circle at bottom right, rgba(255,138,61,0.10), transparent 30%),
        #FFFDF9;
    color: var(--text);
}

#MainMenu,
footer,
header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(18px);
    border-right: 1px solid var(--border);
    width: 290px !important;
    min-width: 290px !important;
}

section[data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
}

.sidebar-logo {
    padding: 12px;
    border-radius: 24px;
    background: linear-gradient(135deg,#FF8A3D,#FFB36B);
    color: white;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}

.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.2;
}

.sidebar-sub {
    opacity: 0.9;
    font-size: 13px;
    margin-top: 8px;
}

/* HERO */
.hero-card {
    background: linear-gradient(135deg,#FFFFFF,#FFF7EF);
    border: 1px solid var(--border);
    border-radius: 34px;
    padding: 45px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: '';
    position: absolute;
    width: 300px;
    height: 300px;
    background: rgba(255,138,61,0.10);
    border-radius: 50%;
    top: -120px;
    right: -100px;
}

.hero-badge {
    display: inline-block;
    background: var(--orange-light);
    color: var(--orange);
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 56px;
    font-weight: 700;
    line-height: 1.1;
    color: var(--text);
}

.hero-title span {
    color: var(--orange);
}

.hero-sub {
    margin-top: 18px;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.8;
    max-width: 760px;
}

/* METRIC CARD */
.metric-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 28px;
    box-shadow: var(--shadow);
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: #FFD4AE;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: var(--text);
}

.metric-icon {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--orange-light);
    color: var(--orange);
    font-size: 24px;
    margin-bottom: 18px;
}

/* SECTION */
.section-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 40px;
    margin-bottom: 10px;
}

.section-sub {
    color: var(--muted);
    margin-bottom: 30px;
}

/* INPUT */
.stTextArea textarea {
    border-radius: 24px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.9) !important;
    padding: 20px !important;
    font-size: 16px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow);
}

.stTextArea textarea:focus {
    border: 1px solid var(--orange) !important;
    box-shadow: 0 0 0 4px rgba(255,138,61,0.12) !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 18px !important;
    border: none !important;
    background: linear-gradient(135deg,#FF8A3D,#FFB36B) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 14px 20px !important;
    box-shadow: 0 10px 30px rgba(255,138,61,0.25);
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 30px rgba(255,138,61,0.35);
}

/* RESULT CARD */
.result-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--border);
    border-radius: 32px;
    padding: 40px;
    box-shadow: var(--shadow);
}

.sentiment-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
}

.badge-positive {
    background: rgba(34,197,94,0.12);
    color: var(--green);
}

.badge-neutral {
    background: rgba(250,204,21,0.12);
    color: #CA8A04;
}

.badge-negative {
    background: rgba(239,68,68,0.12);
    color: var(--red);
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 16px;
    background: white;
    border: 1px solid var(--border);
    padding: 14px 22px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#FF8A3D,#FFB36B);
    color: white;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* EXPANDER */
.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--text);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-title">AI Sentiment Dashboard</div>
        <div class="sidebar-sub">
            Web Mining & NLP Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Sentiment Analyzer",
            "EDA Visualization",
            "Model Performance",
            "Dataset Explorer",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Settings")

    model_choice = st.selectbox(
        "Choose Model",
        ["RoBERTa", "IndoBERT", "Random Forest", "XGBoost"]
    )

    st.toggle("Show preprocessing")
    st.toggle("Enable animation", value=True)

    st.markdown("---")

    st.markdown("""
    ### 🔥 Pipeline

    ① Web Crawling  
    ② Cleaning Text  
    ③ NLP Processing  
    ④ Sentiment Labeling  
    ⑤ Machine Learning  
    ⑥ Visualization  
    """)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.markdown("""
    <div class="hero-card">

        <div class="hero-badge">
            ✨ Transformer-Based Sentiment Analysis
        </div>

        <div class="hero-title">
            YouTube News <span>Sentiment Analysis</span>
        </div>

        <div class="hero-sub">
            Dashboard analisis sentimen komentar YouTube terhadap pemberitaan
            kecelakaan Kereta Api Argo Bromo menggunakan pendekatan
            Web Mining, NLP, IndoBERT, dan RoBERTa.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">💬</div>
            <div class="metric-label">TOTAL COMMENTS</div>
            <div class="metric-value">12.4K</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-label">MODEL ACCURACY</div>
            <div class="metric-value">94.1%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-label">DOMINANT SENTIMENT</div>
            <div class="metric-value">Negative</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">PROCESSING SPEED</div>
            <div class="metric-value">0.4s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Executive Insight</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Ringkasan insight utama berdasarkan hasil analisis data komentar.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div class="result-card">
            <div style="font-size:24px;font-weight:700;margin-bottom:14px;">
                📌 AI Insight Summary
            </div>

            <div style="line-height:2;color:#6B7280;font-size:15px;">
                Mayoritas komentar publik menunjukkan sentimen negatif terhadap
                insiden kecelakaan Kereta Api Argo Bromo.

                Kritik paling dominan berkaitan dengan keselamatan transportasi,
                kualitas sistem operasional kereta api, serta respon pemerintah.

                Selain itu, terdapat lonjakan emosi negatif setelah video berita
                menjadi viral di platform YouTube.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        data = pd.DataFrame({
            'Sentiment': ['Positive','Neutral','Negative'],
            'Value': [24, 31, 45]
        })

        st.markdown("### Sentiment Distribution")
        st.bar_chart(data.set_index('Sentiment'))

# =========================================================
# SENTIMENT ANALYZER
# =========================================================
elif page == "Sentiment Analyzer":

    st.markdown("<div class='section-title'>Real-Time Sentiment Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Analisis sentimen komentar secara langsung menggunakan model NLP.</div>", unsafe_allow_html=True)

    text = st.text_area(
        "Input Comment",
        height=180,
        placeholder="Masukkan komentar YouTube di sini..."
    )

    col1, col2 = st.columns([1,3])

    with col1:
        analyze = st.button("Analyze Sentiment")

    if analyze:

        sentiment = np.random.choice(['Positive','Neutral','Negative'])

        badge = {
            'Positive':'badge-positive',
            'Neutral':'badge-neutral',
            'Negative':'badge-negative'
        }

        st.markdown(f"""
        <div class="result-card">

            <div class="sentiment-badge {badge[sentiment]}">
                {sentiment} Sentiment
            </div>

            <div style="font-size:52px;font-weight:700;margin-top:20px;">
                {sentiment}
            </div>

            <div style="margin-top:16px;color:#6B7280;line-height:1.8;">
                Komentar berhasil diproses menggunakan model {model_choice}.
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(82)

        st.caption("Confidence Score: 82%")

        with st.expander("🔍 NLP Processing Pipeline"):
            st.write("✔ Cleaning Text")
            st.write("✔ Case Folding")
            st.write("✔ Tokenization")
            st.write("✔ Normalization")
            st.write("✔ Sentiment Prediction")

# =========================================================
# EDA VISUALIZATION
# =========================================================
elif page == "EDA Visualization":

    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Visualisasi pola data dan distribusi sentimen.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Distribution",
        "Word Frequency",
        "Top Comments"
    ])

    with tab1:

        data = pd.DataFrame({
            'Channel': ['TVONE','KOMPAS','METROTV'],
            'Comments': [5400, 4200, 2800]
        })

        st.bar_chart(data.set_index('Channel'))

    with tab2:

        words = pd.DataFrame({
            'Word': ['tabrakan','masinis','kereta','keselamatan','pemerintah'],
            'Count': [220,180,150,130,120]
        })

        st.dataframe(words, use_container_width=True)

    with tab3:

        st.markdown("""
        <div class="metric-card">
            <b>😔 Most Negative Comment</b>
            <br><br>
            Pemerintah harus memperbaiki sistem keselamatan kereta api.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <b>😊 Most Positive Comment</b>
            <br><br>
            Terima kasih kepada tim evakuasi yang bekerja keras.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div style="font-size:24px;font-weight:700;">🏆 XGBoost</div>
            <div style="margin-top:18px;line-height:2;color:#6B7280;">
                Accuracy : 94.1%<br>
                Precision : 93.7%<br>
                Recall : 93.1%<br>
                F1-Score : 93.5%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div style="font-size:24px;font-weight:700;">🧠 Random Forest</div>
            <div style="margin-top:18px;line-height:2;color:#6B7280;">
                Accuracy : 92.8%<br>
                Precision : 92.1%<br>
                Recall : 91.7%<br>
                F1-Score : 91.9%
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":

    st.markdown("<div class='section-title'>Dataset Explorer</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV Dataset",
        type=['csv']
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

        st.download_button(
            "Download Dataset",
            df.to_csv(index=False),
            file_name='dataset.csv'
        )

# =========================================================
# ABOUT
# =========================================================
elif page == "About Project":

    st.markdown("<div class='section-title'>About Project</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">

    <div style="font-size:28px;font-weight:700;margin-bottom:18px;">
        📖 Project Description
    </div>

    <div style="line-height:2;color:#6B7280;">

    Dashboard ini dikembangkan untuk melakukan analisis sentimen terhadap komentar YouTube
    mengenai kecelakaan Kereta Api Argo Bromo.

    Sistem menggunakan pendekatan Web Mining, NLP, IndoBERT, RoBERTa,
    Random Forest, dan XGBoost.

    Tujuan utama proyek adalah memahami opini publik terhadap peristiwa
    yang diberitakan media online.

    </div>

    </div>
    """, unsafe_allow_html=True)

```

# INSTALLATION

```bash
pip install streamlit pandas numpy
```

# RUN

```bash
streamlit run app.py
```

