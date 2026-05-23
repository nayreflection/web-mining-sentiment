import streamlit as st

def dashboard_page():

    st.markdown("""
    <div class="hero-title">
        News Sentiment Analysis
        <span class="hero-accent">
            Agro Bromo
        </span>
    </div>

    <div class="hero-sub">
        Analisis sentimen komentar YouTube
        terhadap kecelakaan Kereta Api Agro Bromo
        menggunakan Transformer NLP berbasis
        IndoBERT dan RoBERTa.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">
                TOTAL KOMENTAR
            </div>

            <div class="metric-value">
                3,248
            </div>

            <div class="metric-sub">
                Crawling dari YouTube API
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">
                MODEL NLP
            </div>

            <div class="metric-value" style="color:#FF6B2C;">
                IndoBERT
            </div>

            <div class="metric-sub">
                Transformer Indonesian
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">
                SENTIMEN DOMINAN
            </div>

            <div class="metric-value" style="color:#FF5A5A;">
                NEGATIF
            </div>

            <div class="metric-sub">
                47% dari total komentar
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown("""
    <div class="section-title">
        Channel Sumber Data
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    channels = [
        ("TVONE", "#FF5A5A"),
        ("KOMPAS TV", "#4A9FFF"),
        ("METRO TV", "#35D07F")
    ]

    for col, item in zip([c1,c2,c3], channels):

        name, color = item

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div style="
                    width:12px;
                    height:12px;
                    border-radius:50%;
                    background:{color};
                    margin-bottom:16px;
                "></div>

                <div style="
                    font-size:24px;
                    font-weight:700;
                ">
                    {name}
                </div>

                <div class="metric-sub">
                    Sumber komentar YouTube
                </div>
            </div>
            """, unsafe_allow_html=True)
