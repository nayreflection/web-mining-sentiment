import streamlit as st

from pages.dashboard import dashboard_page
from pages.single_predict import single_predict_page
from pages.statistics import statistics_page
from pages.batch_predict import batch_predict_page
from pages.about import about_page

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# LOAD CSS
# =========================================
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-dot"></div>
        <div>
            <div class="brand-title">
                WEB MINING
            </div>
            <div class="brand-sub">
                SENTIMENT ANALYSIS
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analisis Sentimen",
            "Statistik",
            "Batch Prediksi",
            "Tentang"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        <div>Transformer NLP</div>
        <div style="margin-top:8px;">
            IndoBERT · RoBERTa
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# ROUTING
# =========================================
if page == "Dashboard":
    dashboard_page()

elif page == "Analisis Sentimen":
    single_predict_page()

elif page == "Statistik":
    statistics_page()

elif page == "Batch Prediksi":
    batch_predict_page()

elif page == "Tentang":
    about_page()
