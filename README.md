# Stock Bottleneck

## Scientific Overview

**Stock Bottleneck** is a project designed to perform anomaly detection on stock market data using autoencoder neural networks. The core idea is to identify unusual patterns or outliers in stock price movements, which may indicate significant market events or irregularities.

- **Data Source:** Historical stock data is fetched from Yahoo Finance, including features such as Open, High, Low, Close, Adjusted Close, and Volume.
- **Preprocessing:** Data is forward-filled to handle missing values and normalized to ensure effective neural network training.
- **Model:** The anomaly detection model is a custom autoencoder built with TensorFlow/Keras. It compresses (encodes) the input data into a lower-dimensional latent space and then reconstructs (decodes) it. The model uses a reparameterization trick, similar to variational autoencoders, to improve robustness.
- **Anomaly Detection:** Anomalies are detected by measuring the reconstruction error (difference between original and reconstructed data). Points with errors above a dynamic threshold (mean + 2*std) are flagged as anomalies.
- **Visualization:** The app provides visualizations of the stock data, reconstruction, and detected anomalies, as well as example reconstructions for qualitative assessment.

## Development Overview

### Project Structure

- `main.py`: Streamlit app entry point. Handles user input, data fetching, model training, and visualization.
- `utils.py`: Utility functions for data processing, model creation, training, and plotting.
- `constants.py`: Project-wide constants (dates, model parameters, etc.).
- `train.ipynb`: Jupyter notebook for data exploration and model training experiments.
- `optimize.ipynb`:  Jupyter notebook for hyperparameter tuning and further experiments.
- `Stock_Bottleneck.keras`: Saved Keras model file.
- `report.pdf`: Scientific/technical report.
- `Pipfile`/`Pipfile.lock`: Dependency management for Python environment.

### Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install pipenv
   pipenv install
   ```

2. **Run the Streamlit app:**
   ```bash
   pipenv run streamlit run main.py
   ```

3. **Interact with the app:**
   - Enter a stock ticker (e.g., `AMZN`) to fetch data and run anomaly detection.
   - View metrics, anomaly plots, and reconstruction examples.

### Dependencies

- Python 3.10
- streamlit
- tensorflow
- yfinance
- pandas
- numpy
- matplotlib

(See `Pipfile` for full details.)

### Notes

- The app trains the model on-the-fly for the selected stock ticker.
- For large-scale or production use, consider pre-training and saving models for faster inference.
- The Jupyter notebooks can be used for further experimentation and model development. 