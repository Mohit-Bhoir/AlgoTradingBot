import streamlit as st
import pandas as pd
import json
import os
import subprocess
import signal
import time
import tpqoa
from dotenv import load_dotenv
import yaml
import plotly.graph_objects as go
import sys
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Page Config1
st.set_page_config(page_title="AlgoTrading Bot Dashboard", layout="wide")

st.title("AlgoTrading Bot Live Dashboard")

st.markdown("""
**Note:** This dashboard and trading bot are for illustration and educational purposes only. Actual production trading bots do not typically operate at a 1-minute granularity, and the PnL shown here is not guaranteed to be optimal or representative of real-world trading performance. Use at your own risk.
""")

# ------------- Helper Functions -------------

def load_params():
    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

def get_bot_status():
    if os.path.exists("bot.pid"):
        try:
            with open("bot.pid", "r") as f:
                pid = int(f.read())
            # Check if process exists
            try:
                os.kill(pid, 0)
                return True, pid
            except OSError:
                return False, None
        except:
             return False, None
    return False, None

def start_bot(units):
    if get_bot_status()[0]:
        st.warning("Bot is already running.")
        return

    env = os.environ.copy()
    env["TRADING_UNITS"] = str(units)
    
    # Run the bot as a subprocess with output captured to log
    # remove creationflags=subprocess.CREATE_NEW_CONSOLE for integrated running
    # but we need to run it detached so blocking doesn't happen
    
    # Clear log
    with open("bot.log", "w") as f:
        f.write("Starting bot...\n")

    # Start detached process redirecting stdout/stderr to file
    log_file = open("bot.log", "a")
    
    # Use pythonw if on windows to avoid popping a window, 
    # OR just use standard python with creationflags if we want to HIDE it.
    # To hide console window on Windows:
    if os.name == 'nt':
        # CREATE_NO_WINDOW = 0x08000000
        # DETACHED_PROCESS = 0x00000008
        creation_flags = 0x08000000 
    else:
        creation_flags = 0

    process = subprocess.Popen([sys.executable, "-u", "src/live_stream.py"], env=env, 
                               cwd=os.getcwd(), 
                               stdout=log_file, 
                               stderr=log_file,
                               creationflags=creation_flags)
    
    with open("bot.pid", "w") as f:
        f.write(str(process.pid))
    
    st.success(f"Bot started with PID {process.pid}")
    time.sleep(1)
    st.rerun()

def stop_bot():
    running, pid = get_bot_status()
    if running and pid:
        try:
            os.kill(pid, signal.SIGTERM) # On Windows this might need force kill
        except:
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
        
        if os.path.exists("bot.pid"):
            os.remove("bot.pid")
        st.success("Bot stopped.")
        time.sleep(1)
        st.rerun()
    else:
        st.warning("Bot is not running.")

def get_oanda_credentials():
    load_dotenv()
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_ACCESS_TOKEN")
    account_type = os.environ.get("OANDA_ACCOUNT_TYPE")
    if account_id and access_token and account_type:
        import configparser
        config = configparser.ConfigParser()
        config['oanda'] = {
            'account_id': account_id,
            'access_token': access_token,
            'account_type': account_type
        }
        with open("oanda_temp.cfg", "w") as f:
            config.write(f)
        return "oanda_temp.cfg"
    return None

def get_account_summary():
    conf_path = get_oanda_credentials()
    if not conf_path:
        return {"error": "Oanda credentials not found in .env file."}
    try:
        api = tpqoa.tpqoa(conf_path)
        return api.get_account_summary()
    except Exception as e:
        return {"error": str(e)}

def get_transactions():
    conf_path = get_oanda_credentials()
    if not conf_path:
        return pd.DataFrame()
    try:
        api = tpqoa.tpqoa(conf_path)
        trans = api.get_transactions(tid=0)
        return pd.DataFrame(trans)
    except Exception as e:
        return pd.DataFrame()

def exit_trade():
    """
    Logic to exit/close all open positions.
    Replace this placeholder with actual broker API call.
    """
    # Example: Call your broker's API to close all open positions
    # For now, just log/print
    print("Exit Trade executed: All open positions should be closed.")
    # Optionally, add logging or Streamlit notification
    import logging
    logging.info("Exit Trade executed: All open positions should be closed.")
    return True

# ------------- Sidebar Controls -------------

st.sidebar.header("Bot Controls")

params = load_params()

# Capital Management (Units)
current_units = st.sidebar.number_input("Trading Units (Capital per Trade)", min_value=1000, value=100000, step=1000)


# --- Exit Trade Button ---
if st.sidebar.button("Exit Trade", help="Close all open positions. Bot can be running or stopped."):
    result = exit_trade()
    st.sidebar.success("Exit Trade executed. All open positions should be closed.")

running, pid = get_bot_status()
if running:
    st.sidebar.success(f"Bot Running (PID: {pid})")
    if st.sidebar.button("Stop Bot"):
        stop_bot()
else:
    st.sidebar.error("Bot Stopped")
    if st.sidebar.button("Start Bot"):
        start_bot(current_units)

# ------------- Main Dashboard -------------

# 1. Account Info
st.subheader("Live Account Summary (Oanda)")
conf_path = get_oanda_credentials()
if conf_path and os.path.exists(conf_path):
    summary = get_account_summary()
    if "error" not in summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Balance", f"{summary.get('balance', 0)}")
        col2.metric("P&L", f"{summary.get('pl', 0)}")
        col3.metric("NAV", f"{summary.get('NAV', 0)}")
        col4.metric("Unrealized PL", f"{summary.get('unrealizedPL', 0)}")
    else:
        st.error(f"Could not fetch account data: {summary['error']}")
else:
    st.error(f"Config file not found at {conf_path}")


# 2. Live Chart
st.subheader("Live Market Data & Trades")
stream_file = os.path.join("data", "streamed", "streamed_data.csv")
if os.path.exists(stream_file):
    try:
        df = pd.read_csv(stream_file)
        if not df.empty:
            df['time'] = pd.to_datetime(df.iloc[:, 0]) # Assuming index is time
            df.set_index('time', inplace=True)
            
            # Simple Line for now since we only stream 'c' (close)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['c'], mode='lines', name='Price'))
            
            # Add markers for trades if available
            trades_file = os.path.join("data", "streamed", "trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    t_data = json.load(f)
                if t_data:
                    t_df = pd.DataFrame(t_data)
                    t_df['time'] = pd.to_datetime(t_df['time'])
                    
                    buy_df = t_df[t_df['type'] == 'GOING LONG']
                    sell_df = t_df[t_df['type'] == 'GOING SHORT']
                    
                    fig.add_trace(go.Scatter(x=buy_df['time'], y=buy_df['price'], mode='markers', marker=dict(color='green', size=10), name='Buy'))
                    fig.add_trace(go.Scatter(x=sell_df['time'], y=sell_df['price'], mode='markers', marker=dict(color='red', size=10), name='Sell'))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for data stream...")
    except Exception as e:
        st.error(f"Error reading stream data: {e}")
else:
    st.info("No streamed data found. Start the bot to generate data.")

# 3. Live Logs & Ticker
st.subheader("Live Market Status")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Current Tick & Status")
    tick_file = os.path.join("data", "streamed", "latest_tick.json")
    if os.path.exists(tick_file):
        try:
            with open(tick_file, 'r') as f:
                tick_data = json.load(f)
            
            # Formatting Position
            pos = tick_data.get('position', 0)
            if pos == 1:
                pos_str = "LONG 🟢"
            elif pos == -1:
                pos_str = "SHORT 🔴"
            else:
                pos_str = "NEUTRAL ⚪"
            
            # Formatting PnL
            cum_pnl = tick_data.get('cum_pnl', 0.0)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"{tick_data['price']:.5f}")
            m2.metric("Position", pos_str)
            m3.metric("Cum. PnL", f"{cum_pnl:.2f}")

            st.text(f"Bid: {tick_data['bid']} | Ask: {tick_data['ask']}")
            st.text(f"Time: {tick_data['time']}")
        except Exception as e:
             st.text(f"Reading Tick Data... {e}")
    else:
        st.text("No Tick Data")

with col2:
    st.markdown("### Bot Logs")
    if os.path.exists("bot.log"):
        with open("bot.log", "r") as f:
            lines = f.readlines()
            # Show last 10 lines
            log_content = "".join(lines[-20:])
            st.code(log_content, language="text")
    else:
        st.info("No logs found.")

# 4. Recent Trades
st.subheader("Recent Bot Trades")
trades_file = os.path.join("data", "streamed", "trades.json")

if os.path.exists(trades_file):
    try:
        with open(trades_file, 'r') as f:
            trades_data = json.load(f)
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
            # Remove 'pl' and 'cum_pl' columns if present
            trades_df = trades_df.drop(columns=[col for col in ['pl', 'cum_pl'] if col in trades_df.columns], errors='ignore')
            st.dataframe(trades_df.sort_index(ascending=False))
        else:
            st.info("No trades recorded locally yet.")
    except:
        st.info("Error reading local trade log.")
else:
    st.info("No local trade history.")
    
# 4. Oanda Transactions
st.subheader("Oanda Transaction Report")
if st.button("Refresh Oanda Transactions"):
    trans_df = get_transactions()
    if not trans_df.empty:
        # Drop columns that are all NaN or all None
        trans_df = trans_df.dropna(axis=1, how='all')
        trans_df = trans_df.loc[:, trans_df.apply(lambda col: not all(x is None for x in col))]
        # Show only the 5 most recent rows
        st.dataframe(trans_df.tail(5))
    else:
        st.info("No transactions found or error fetching.")

st.markdown(
    """
    <hr style="margin-top:40px;margin-bottom:10px; border-color: #333;">
    <div style="text-align:center; font-size:14px; color:#fff;">
        <strong>Forex Algotrading Bot</strong> &middot; v0.1.0 &middot; Demo Build by Mohit Bhoir<br>
        Data Sources: <a href="https://www.oanda.com/" target="_blank" style="color:#ffd700;">OANDA API</a><br>
        <span style="color:gold; font-size:18px;">&#11088;</span>
        <a href="https://github.com/Mohit-Bhoir/AlgoTradingBot" target="_blank" style="color:#ffd700;">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
# Auto-refresh log
if running:
    time.sleep(5)
    st.rerun()
