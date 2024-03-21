"""
Main file for the Stock Bottleneck App.
"""

import streamlit as st

st.set_page_config(
    page_title="Stock Bottleneck",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {display: none;}
footer {display: none;}
.stDeployButton {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Stock Bottleneck")
st.subheader("Stock Anomaly Detection App using Autoencoder Neural Network")
