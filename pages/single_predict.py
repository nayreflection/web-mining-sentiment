import streamlit as st

def show_single_predict():

    st.title("Single Prediction")

    text = st.text_area("Masukkan komentar")

    if st.button("Analisis"):
        st.success("Hasil sentimen muncul di sini")
