# Nifty 50 Market Regime Detection using Gaussian Mixture Models (GMM)

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/YOUR_USERNAME/YOUR_REPO/blob/main/nifty50_regime_gmm_polished.ipynb)
*(Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub repository details to enable the nbviewer link.)*

This is a complete, focused Quantitative Machine Learning project that detects the "mood" or "regime" of the Indian Stock Market (NSE Nifty 50) on any given trading day. The model automatically classifies each day into a **Bull (Uptrend)**, **Bear (Downtrend)**, or **Sideways (Ranging)** regime using unsupervised machine learning.

Instead of trying to predict future stock prices—which is highly inaccurate—this project solves an industry-standard problem: dynamically understanding what state the market is currently in, so quantitative trading strategies can adapt their risk accordingly.

---

## 📈 Key Concepts Used

### 1. Gaussian Mixture Model (GMM) vs K-Means
Most beginner clustering projects use K-Means. This project uses **Gaussian Mixture Models (GMM)**, which is significantly more realistic for financial data:
* **Soft Probabilistic Assignments:** K-Means assigns a "hard label" (saying a day is 100% Bull). GMM assigns a "soft label" (saying a day is 70% Bull, 25% Sideways, 5% Bear). This beautifully captures the uncertainty and transition phases of financial markets.
* **Unique Covariance (Shapes):** Bear markets act very differently than Bull markets (they are highly volatile and spread out). GMM allows each regime to have its own unique shape (covariance), whereas K-Means assumes all regimes act exactly the same.

### 2. Feature Engineering (Quant Indicators)
We construct 4 highly specific financial features to feed into the model, avoiding "black box" machine learning:
1. **Log Returns:** We calculate logarithmic daily returns instead of simple percentage returns, as log returns are mathematically symmetric and additive over time.
2. **20-Day Rolling Volatility:** The standard deviation of log returns over a one-month window, capturing how "jumpy" or fearful the market is.
3. **RSI (Relative Strength Index - 14d):** A standard momentum oscillator that bounds the speed and change of price movements between 0 and 100.
4. **Moving Average Ratio (50d / 200d):** Compares short-term market trends against the long-term trend (useful for identifying structural shifts like a "Golden Cross").

### 3. Model Evaluation Without Labels (BIC & Silhouette)
Because this is Unsupervised Learning, there are no "correct answers" to grade the model against. Instead, we use:
* **Bayesian Information Criterion (BIC):** A statistical score used to mathematically prove that **K=3** (3 market regimes) is the optimal number of clusters by balancing model fit against complexity.
* **Silhouette Score:** Evaluates how well separated the Bull, Bear, and Sideways clusters are.
* **Historical Validation:** We visually map the model's detected Bear regimes over the last 10 years and confirm they perfectly align with known Indian market crashes (e.g., the March 2020 COVID-19 crash, the 2018 IL&FS crisis).

---

## 💻 Tech Stack
* **Python 3**
* `yfinance` (Live historical data extraction)
* `pandas` & `numpy` (Data manipulation & Feature engineering)
* `scikit-learn` (StandardScaler, GaussianMixture model, Metrics)
* `matplotlib` & `seaborn` (Financial data visualization)

---

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install yfinance pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open the `nifty50_regime_gmm_polished.ipynb` notebook via Jupyter or upload it directly to **Google Colab**.
4. Run all cells to see the Nifty 50 data download in real-time, the model train, and the generation of all the regime charts.

---

### Project Results Summary (2014-2024 Data)
The model naturally identified the following segments mathematically, validating the financial theory:
* **Bull Regime (62.1% of days):** Highest average log return (+0.07%), lowest volatility (0.67%).
* **Bear Regime (2.8% of days):** Extremely negative average return (-0.36%), massive volatility (3.22%). Occurred entirely during major crashes.
* **Sideways Regime (35.0% of days):** Near-zero returns (+0.02%), moderate volatility (1.15%).

*Note: The model correctly flags **Transition Days** (~5.7% of the dataset) where the market is rapidly changing, and the model's maximum certainty drops below 60%.*
