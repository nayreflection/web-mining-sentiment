import streamlit as st

def show_dashboard():

    st.markdown("""
    <div class="hero-title">
        News Sentiment Analysis
        <span class="hero-accent">Agro Bromo</span>
    </div>

    <div class="hero-sub">
        Analisis sentimen komentar YouTube
        menggunakan IndoBERT dan RoBERTa
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Data</h3>
            <h1>3,200</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Model</h3>
            <h1>IndoBERT</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Sentimen Dominan</h3>
            <h1 style="color:#FF6B2C;">Negatif</h1>
        </div>
        """, unsafe_allow_html=True)
