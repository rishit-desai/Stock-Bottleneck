"""
Utility functions to be used in the main script.
"""

# pylint: disable-all

# from datetime import datetime
# from dateutil.relativedelta import relativedelta
import streamlit as st
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import constants as consts
import matplotlib.pyplot as plt
import numpy as np


def get_dates():
    # Dynamic dates
    # date = datetime.now() - relativedelta(years=years)
    # return date.strftime("%Y-%m-%d")

    # Static dates
    return consts.START_DATE, consts.END_DATE


def info_is_available(info, key) -> bool:
    if key in info and info[key] != None:
        return True

    return False


@st.cache_data(ttl=3600)
def get_data(stock: str):
    """
    Get preprocessed stock data from Yahoo Finance.
    """
    start, end = get_dates()
    data = yf.download(stock, start=start, end=end)
    data = data.ffill()
    data_normalized = (data - data.min()) / (data.max() - data.min())
    return data_normalized


def _encoder(latent_dims=16, input=keras.layers.Input(shape=(6,))):
    x = keras.layers.Dense(64, activation="relu")(input)
    x = keras.layers.Dense(32, activation="relu")(x)
    mu = keras.layers.Dense(latent_dims, name="latent_mu")(x)
    sigma = keras.layers.Dense(latent_dims, name="latent_sigma")(x)
    return mu, sigma


def _reparameterize(mean, log_var):
    def sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=(K.shape(mean)))
        return mean + K.exp(0.5 * log_var) * epsilon

    return keras.layers.Lambda(sampling)([mean, log_var])


def _decoder(latent_inputs, input_dims=6):
    x = keras.layers.Dense(32, activation="relu")(latent_inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(input_dims, activation="tanh")(x)
    return outputs


def _create_model(
    input_dim=6,
    latent_dims=consts.LATENT_DIMS,
    loss=consts.LOSS,
    optimizer=consts.OPTIMIZER,
):
    model_input = keras.layers.Input(shape=(input_dim,))
    mu, sigma = _encoder(input=model_input, latent_dims=latent_dims)
    latent = _reparameterize(mu, sigma)
    decoded = _decoder(latent_inputs=latent, input_dims=input_dim)
    model = keras.Model(inputs=model_input, outputs=decoded)

    model.compile(optimizer=optimizer, loss=loss)

    return model


def _fit_model(model, data, epochs=consts.EPOCHS, batch_size=consts.BATCH_SIZE):
    training_bar = st.progress(0, text=consts.TRAINING_PROGRESS_TEXT)
    bar_updater = _UpdateProgressBar(training_bar, epochs=epochs)
    model.fit(
        data,
        data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[bar_updater],
    )
    training_bar.progress(100)


class _UpdateProgressBar(Callback):
    def __init__(self, bar, epochs=consts.EPOCHS):
        self.bar = bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        self.bar.progress(epoch / self.epochs, text=consts.TRAINING_PROGRESS_TEXT)


@st.cache_resource(ttl=3600, show_spinner=False)
def get_model(stock: str):
    model = _create_model()

    data = get_data(stock=stock)

    _fit_model(model, data)

    return model


def plot_reconstruction(model, train_data):
    reconstructed_data = model.predict(train_data)
    reconstruction_errors = np.mean(np.square(reconstructed_data - train_data), axis=1)

    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

    fig, ax = plt.subplots(figsize=(10, 6))

    plt.plot(reconstruction_errors)
    ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Time")
    plt.ylabel("Reconstruction Error")
    plt.title("Anomaly Detection with Threshold")
    plt.legend()

    st.pyplot(fig)


def plot_examples(train_data, reconstructed_data):

    stock_decoded = reconstructed_data[:, 3]
    stock_input = train_data.values[:, 3]

    stock_windows = []
    decoded_windows = []
    for _ in range(10):
        start = np.random.randint(0, len(stock_input) - 200)
        window = stock_input[start : start + 200]
        stock_windows.append(window)
        window = stock_decoded[start : start + 200]
        decoded_windows.append(window)

    plt.figure(figsize=(10, 30))

    for i in range(10):
        plt.subplot(8, 2, i + 1)
        plt.plot(stock_windows[i], label="Stock Data")
        plt.plot(decoded_windows[i], label="Decoded Data")
        plt.title("Example " + str(i + 1))
        plt.legend()

    st.pyplot(plt)
