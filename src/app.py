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
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000)
# Confidence or other params could go here if model outputs probs

# --- ITERATIVE BACKTEST EXECUTION ---
if st.button("Run Iterative Backtest"):
    from IterativeBacktest import IterativeBacktest
    
    # 1. Init Backtest Engine (loads data internally from processed/data_clean.csv)
    # We pass the same data used by the pipeline
    bc = IterativeBacktest(
        symbol="EUR_USD", 
        start=None, 
        end=None, 
        amount=initial_capital, 
        use_spread=True,
        data_path=data_path
    )
    
    # 2. Run Strategy using the loaded model
    with st.spinner("Running Event-Driven Backtest..."):
        bc.test_xgboost_strategy(model)
    
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
    else:
        st.warning("No trades executed or no history available.")

st.divider()

# --- Layout (Legacy Vectorized) ---

col_1, col_2 = st.columns([3, 1])

with col_1:
    st.subheader("Vectorized Backtest (Reference)")

# Prepare data for prediction (Legacy Vectorized)
X = data.copy()
# Drop target/meta columns as done in train/evaluate
if "direction" in X.columns:
    X = X.drop(columns=["direction"])
cols_to_drop = [c for c in ["time", "price", "returns"] if c in X.columns]
X = X.drop(columns=cols_to_drop)

# Make Predictions
try:
    predictions = model.predict(X)
    data['prediction'] = predictions
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.stop()

# --- Backtest Logic ---
# 1 = Buy (Long), 0 = Sell (Short)
# Note: This is an approximation. 
# Robust backtesting needs vectorbt or backtrader. 
# Here we do a simple vectorized backtest.

data['position'] = np.where(data['prediction'] == 1, 1, -1)
data['strategy_returns'] = data['position'] * data['returns']

# Cumulative Returns
data['creturns'] = data['returns'].cumsum().apply(np.exp)
data['cstrategy'] = data['strategy_returns'].cumsum().apply(np.exp)

# Equity Curve
data['equity_bh'] = initial_capital * data['creturns']
data['equity_strategy'] = initial_capital * data['cstrategy']

# --- Layout ---

col_1, col_2 = st.columns([3, 1])

with col_1:
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['equity_bh'], name='Buy & Hold', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=data['time'], y=data['equity_strategy'], name='Strategy', line=dict(color='blue', width=2)))
    fig.update_layout(xaxis_title='Date', yaxis_title='Equity ($)', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col_2:
    st.subheader("Metrics")
    total_return = (data['equity_strategy'].iloc[-1] / initial_capital) - 1
    bh_return = (data['equity_bh'].iloc[-1] / initial_capital) - 1
    
    st.metric("Strategy Return", f"{total_return:.2%}")
    st.metric("Buy & Hold Return", f"{bh_return:.2%}")
    st.metric("Final Equity", f"${data['equity_strategy'].iloc[-1]:,.2f}")

st.subheader("Visual Analysis (Last 500 bars)")
# Zoom in on recent price action
subset = data.tail(500)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=subset['time'], y=subset['price'], name='Price', line=dict(color='black', width=1)))

# Buy Signals (prediction=1)
# Look for crossovers or just raw signals?
# Let's plot points where prediction == 1
buy_idx = subset[subset['prediction'] == 1]
sell_idx = subset[subset['prediction'] == 0]

fig2.add_trace(go.Scatter(x=buy_idx['time'], y=buy_idx['price'], mode='markers', name='Long', marker=dict(color='green', symbol='triangle-up', size=8)))
fig2.add_trace(go.Scatter(x=sell_idx['time'], y=sell_idx['price'], mode='markers', name='Short', marker=dict(color='red', symbol='triangle-down', size=8)))

fig2.update_layout(xaxis_title='Date', yaxis_title='Price', template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("View Raw Data"):
    st.dataframe(data)
