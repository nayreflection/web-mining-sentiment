import streamlit as st

# import halaman
from pages.dashboard import show_dashboard
from pages.single_predict import show_single_predict
from pages.statistics import show_statistics
from pages.batch_predict import show_batch_predict
from pages.about import show_about

# CONFIG
st.set_page_config(
    page_title="News Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        WEB MINING <br> SENTIMENT
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Menu",
        [
            "Dashboard",
            "Analisis Sentimen",
            "Statistik",
            "Batch Prediksi",
            "Tentang"
        ]
    )

# ROUTING
if page == "Dashboard":
    show_dashboard()

elif page == "Analisis Sentimen":
    show_single_predict()

elif page == "Statistik":
    show_statistics()

elif page == "Batch Prediksi":
    show_batch_predict()

elif page == "Tentang":
    show_about()
