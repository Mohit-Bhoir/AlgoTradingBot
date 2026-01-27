# AlgoTradingLive 📈🤖

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)![DVC](https://img.shields.io/badge/DVC-enabled-purple.svg)![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red.svg)![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> **A Real-Time Machine Learning Algorithmic Trading Bot with StreamlitDashboard, Oanda Integration, and MLOps Support**

AlgoTradingLive is a full-stack quantitative trading system for Forexmarkets (e.g., EUR/USD) that combines:

-   📡 Live market streaming from Oanda
-   🧠 XGBoost-based predictive modeling
-   ⚡ Automated trade execution
-   📊 Real-time monitoring dashboard
-   🔁 Reproducible MLOps pipelines with DVC & MLflow

---

## 🚀 Features

-   **Live Data Streaming** --- Real-time ingestion via Oanda API
-   **ML-Driven Trading** --- Predicts LONG/SHORT moves
-   **Automated Execution** --- Executes trades programmatically
-   **Trade Logging** --- Stores every trade and P&L snapshot
-   **Dashboard** --- Streamlit app for monitoring performance
-   **MLOps** --- Versioned data + experiments
-   **Reproducibility** --- Entire pipeline tracked through DVC

---

## 📁 Project Structure

```
.├── data/│   ├── raw/            # Historical data│   ├── streamed/       # Live streamed data and trade logs│   └── processed/     # Cleaned/merged data for ML│├── models/             # Trained ML models│├── src/│   ├── app.py          # Streamlit dashboard│   ├── live_stream.py # Main trading bot logic│   ├── train.py       # Model training script│   ├── preprocess.py # Data preprocessing│   └── merge_data.py # Merges historical and live data│├── params.yaml         # Pipeline and model parameters├── requirements.txt   # Python dependencies├── dvc.yaml            # DVC pipeline definition└── README.md
```

---

## 📊 Data & Model Flow

1.  Fetch historical data from Oanda
2.  Stream live prices → `streamed_data.csv`
3.  Merge historical + live data
4.  Preprocess for ML
5.  Train XGBoost model
6.  Log metrics to MLflow
7.  Deploy for live trading

---

## 🖥️ Dashboard

The Streamlit UI shows:

-   Current price & position
-   Cumulative P&L
-   Recent bot trades
-   Oanda transactions
-   Bot health status

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AlgoTradingLive.gitcd AlgoTradingLive
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venvsource venv/bin/activate   # Linux / MacvenvScriptsactivate    # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OANDA_API_KEY=your_api_keyOANDA_ACCOUNT_ID=your_account_idOANDA_ENV=practice
```

---

## 🔁 Run the DVC Pipeline

Initialize DVC:

```bash
dvc init
```

Pull tracked data/models:

```bash
dvc pull
```

Reproduce the pipeline:

```bash
dvc repro
```

Show pipeline DAG:

```bash
dvc dag
```

View metrics:

```bash
dvc metrics show
```

---

## 📈 Train the Model Manually

```bash
python src/train.py
```

---

## 📡 Start Live Trading Bot

```bash
python src/live_stream.py
```

---

## 🖥️ Launch Dashboard

```bash
streamlit run src/app.py
```

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.
> 
> It is **not financial advice**. Trading Forex involves substantialrisk.
> 
> The authors assume no responsibility for trading losses.

---

## ⭐ Star This Repo

If you find this useful for learning quant trading, ML pipelines, orreal-time systems --- drop a ⭐ and follow along for updates 🚀📊

# ---
## 🔑 Oanda Credentials Setup (NEW)

This project now uses a `.env` file for secure and easy credential management. You **must** create a `.env` file in your project root with the following content:

```
OANDA_ACCOUNT_ID="your_account_id"
OANDA_ACCESS_TOKEN="your_access_token"
OANDA_ACCOUNT_TYPE="practice"  # or "live"
```

- Never commit your `.env` file to public repositories.
- All scripts and the dashboard will automatically load credentials from `.env`.

---