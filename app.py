import streamlit as st

# import halaman
from pages.dashboard import show_dashboard
from pages.single_predict import show_single_predict
from pages.batch_predict import show_batch_predict
from pages.statistics import show_statistics
from pages.about import show_about

# ── CONFIG ─────────────────────────────
st.set_page_config(
    page_title="News Sentiment Analysis",
    layout="wide"
)

# ── LOAD CSS ───────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────
st.sidebar.title("📊 Sentiment Analysis")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Single Predict",
        "Batch Predict",
        "Statistics",
        "About"
    ]
)

# ── NAVIGASI ───────────────────────────
if menu == "Dashboard":
    show_dashboard()

elif menu == "Single Predict":
    show_single_predict()

elif menu == "Batch Predict":
    show_batch_predict()

elif menu == "Statistics":
    show_statistics()

elif menu == "About":
    show_about()
