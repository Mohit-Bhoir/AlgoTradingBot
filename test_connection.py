import tpqoa
import os

try:
    conf_path = "API CONNECT/oanda.cfg"
    if not os.path.exists(conf_path):
        print(f"Error: Config file not found at {conf_path}")
    else:
        api = tpqoa.tpqoa(conf_path)
        print("Successfully connected to Oanda.")
        
        # Check account summary
        summary = api.get_account_summary()
        print(f"Account Balance: {summary['balance']}")
        print(f"Account ID: {api.account_id}")
        
        # Check open positions
        # Note: tpqoa might not have get_positions, checking dir
        # print("Methods:", dir(api))
        
        # Try getting history to confirm data access
        data = api.get_history(instrument="EUR_USD", start="2025-01-01", end="2025-01-02", granularity="H1", price="M")
        print(f"Historical data fetch test: {len(data)} rows retrieved.")

        print("\n--- Diagnostic Info ---")
        print(f"Connected Account ID: {api.account_id}")
        print(f"Current Balance: {summary['balance']}")
        # Try to infer if it's practice from ID (usually starts with 101 for practice/v20)
        env_type = "Practice/Demo" if api.account_id.startswith("101") else "Live"
        print(f"Account Type (inferred): {env_type}")


except Exception as e:
    print(f"Connection failed: {e}")
