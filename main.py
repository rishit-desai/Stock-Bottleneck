"""
Main file for the Stock Bottleneck App.
"""

import streamlit as st
from utils import get_model, info_is_available
from constants import HIDE_STREAMLIT_STYLE
import yfinance as yf


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

company_info = yf.Ticker(st.session_state.stock).info

info_container = st.container()

col1, col2, col3 = info_container.columns(3)

with col1:
    st.metric(
        label=company_info["shortName"],
        value="%.2f" % company_info["currentPrice"],
        delta="%.2f" % (company_info["currentPrice"] - company_info["previousClose"]),
    )
with col2:
    st.metric(
        label="Today's High",
        value=(
            "%.2f" % company_info["dayHigh"]
            if (info_is_available(company_info, "dayHigh"))
            else "N/A"
        ),
    )
with col3:
    st.metric(
        label="Today's Low",
        value=(
            "%.2f" % company_info["dayLow"]
            if (info_is_available(company_info, "dayLow"))
            else "N/A"
        ),
    )

col4, col5, col6 = info_container.columns(3)

with col4:
    st.metric(
        label="Revenue Growth (YoY)",
        value=(
            "%.2f" % (company_info["revenueGrowth"] * 100) + "%"
            if (info_is_available(company_info, "revenueGrowth"))
            else "N/A"
        ),
    )
with col5:
    st.metric(
        label="PE Ratio",
        value=(
            "%.2f" % company_info["trailingPE"]
            if (info_is_available(company_info, "trailingPE"))
            else "N/A"
        ),
    )
with col6:
    st.metric(
        label="PB Ratio",
        value=(
            "%.2f" % company_info["priceToBook"]
            if (info_is_available(company_info, "priceToBook"))
            else "N/A"
        ),
    )


model = get_model(stock=st.session_state.stock)

st.text("Model Training Complete!")
