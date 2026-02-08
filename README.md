# 🇮🇳 NiftySentinel AI: Python-Powered Intraday Trading Terminal
> **Automated Machine Learning & Technical Analysis Terminal for Indian Equities (NSE)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market: NSE](https://img.shields.io/badge/Market-NSE_India-orange.svg)](https://www.nseindia.com/)
[![GUI: PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

---

## 📖 Overview
**NiftySentinel AI** is a professional-grade paper trading terminal tailored for the Indian Stock Market. It leverages a **Gradient Boosting Machine Learning model** alongside a suite of technical indicators to automate decision-making for Nifty 50 stocks. 

The terminal provides a real-time, high-performance GUI to monitor price action, AI confidence levels, and automated trade execution within a simulated environment.



---

## 🚀 Core Features
* **AI-Powered Predictions:** Real-time directional bias forecasting using `GradientBoostingClassifier`.
* **Comprehensive Indicators:** Auto-calculates RSI, MACD, Bollinger Bands, ADX, and OBV using `pandas_ta`.
* **Smart Risk Management:**
    * **1.5% Risk Rule:** Automatically calculates quantity based on stop-loss distance.
    * **Trailing Stop-Loss:** Dynamic ATR-based exits to protect profits.
    * **Trade Throttling:** Built-in cooldowns and daily trade limits.
* **Indian Market Native:** Native support for IST market hours (**09:15 - 15:30**) and NSE symbols.
* **State Persistence:** Saves your portfolio, capital, and trade history to a local `.pkl` file.

---

## 🛠️ Technical Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **GUI Framework** | PyQt5 (Fusion Style) |
| **Data Engine** | `yfinance` & `Pandas` |
| **Machine Learning** | `Scikit-Learn` (Gradient Boosting) |
| **Visualization** | `Matplotlib` (Qt5 Backend) |
| **Indicators** | `pandas_ta` |

---

## 📈 Strategy & Scoring Engine
The terminal operates on a **weighted scoring system**. A trade is only considered when multiple signals align:

* **Bullish Score ($\ge +3$):** Triggered by RSI oversold, Bullish MACD cross, and ML model confirmation.
* **Bearish Score ($\le -3$):** Triggered by overbought conditions, bearish volume flow, or ML reversal signals.
* **Daily Trend:** Includes Fibonacci Pivot Points (R1, S1, Pivot) for intraday target setting.

---

## 📦 Installation & Usage

### 1. Requirements
Ensure you have Python installed. Install all dependencies via pip:
```bash
pip install numpy pandas yfinance pandas_ta matplotlib PyQt5 scikit-learn
