"""
Main file for the Stock Bottleneck App.
"""

import streamlit as st
from utils import get_model
from constants import HIDE_STREAMLIT_STYLE

st.set_page_config(
    page_title="Stock Bottleneck",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("Stock Bottleneck")
st.subheader("Stock Anomaly Detection App using Autoencoder Neural Network")

st.text_input(label="Enter Stock Ticker", key="stock", value="AMZN")

model = get_model(stock=st.session_state.stock)

st.text("Model Training Complete!")
