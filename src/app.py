import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import pickle
import os
import numpy as np

# Load Configuration
try:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    st.error("params.yaml not found!")
    st.stop()

st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")
st.title("Algo Trading Backtest Dashboard")

# Paths from params
data_path = params['preprocess']['output']
model_path = params.get('train', {}).get('model_path') # flexible get

if not model_path:
    # Fallback or error
    model_path = "models/model.pkl"

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    return df

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

data = load_data(data_path)
model = load_model(model_path)

if data is None:
    st.error(f"Data not found at {data_path}. Please run `dvc repro`.")
    st.stop()

if model is None:
    st.error(f"Model not found at {model_path}. Please run `dvc repro`.")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    ["XGBoost (ML)", "SMA Crossover", "Bollinger Bands", "Contrarian"]
)

initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000)

# Strategy specific params
if strategy_type == "XGBoost (ML)":
    st.sidebar.info("XGBoost Model (Time Filter 12-16 UTC)")
elif strategy_type == "SMA Crossover":
    sma_s = st.sidebar.number_input("Short SMA", value=50, min_value=1)
    sma_l = st.sidebar.number_input("Long SMA", value=200, min_value=1)
elif strategy_type == "Bollinger Bands":
    bb_sma = st.sidebar.number_input("SMA Window", value=20, min_value=1)
    bb_dev = st.sidebar.number_input("Deviations", value=2, min_value=1)
elif strategy_type == "Contrarian":
    con_window = st.sidebar.number_input("Lookback Window", value=1, min_value=1)

# --- ITERATIVE BACKTEST EXECUTION ---
if st.button("Run Iterative Backtest"):
    from IterativeBacktest import IterativeBacktest
    
    # 1. Init Backtest Engine
    bc = IterativeBacktest(
        symbol="EUR_USD", 
        start=None, 
        end=None, 
        amount=initial_capital, 
        use_spread=True,
        data_path=data_path
    )
    
    # 2. Run selected strategy
    with st.spinner(f"Running {strategy_type} Backtest..."):
        if strategy_type == "XGBoost (ML)":
            bc.test_xgboost_strategy(model)
        elif strategy_type == "SMA Crossover":
            bc.test_sma_strategy(sma_s, sma_l)
        elif strategy_type == "Bollinger Bands":
            bc.test_boll_strategy(bb_sma, bb_dev)
        elif strategy_type == "Contrarian":
            bc.test_con_strategy(con_window)
    
    # 3. Process Results
    history_df = pd.DataFrame(bc.history)
    
    if not history_df.empty:
        history_df = history_df.set_index("time")

        
        col1, col2 = st.columns([3,1])
        
        with col1:
            st.subheader("Net Equity Curve (Inc. Costs)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df.index, 
                y=history_df['equity'], 
                name='Strategy Net', 
                line=dict(color='green', width=2)
            ))
            fig.update_layout(xaxis_title="Time", yaxis_title="Equity ($)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Performance")
            final_equity = history_df['equity'].iloc[-1]
            ret = (final_equity - initial_capital) / initial_capital
            
            st.metric("Final Equity", f"${final_equity:,.2f}")
            st.metric("Total Return", f"{ret:.2%}")
            st.metric("Trades Executed", bc.trades)
            
        with st.expander("View Trade History / Raw Data"):
            st.dataframe(history_df)

    else:
        st.warning("No trades executed or no history available.")

