import pandas as pd
import os
import sys

def merge_datasets():
    # Paths
    raw_path = os.path.join("data", "raw", "data.csv")
    streamed_path = os.path.join("data", "streamed", "streamed_data.csv")
    output_path = os.path.join("data", "processed", "merged_data.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Raw Data
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path, parse_dates=['time'], index_col='time')
    else:
        df_raw = pd.DataFrame()

    # Load Streamed Data
    if os.path.exists(streamed_path):
        try:
            # Try reading with index_col=0 (first column as index)
            df_streamed = pd.read_csv(streamed_path, index_col=0)
            df_streamed.index = pd.to_datetime(df_streamed.index, utc=True)
            df_streamed.index.name = 'time'
        except Exception as e:
            print(f"Error reading streamed data: {e}")
            df_streamed = pd.DataFrame()
    else:
        df_streamed = pd.DataFrame()

    # Ensure raw data index is also UTC aware if streamed is UTC
    if not df_raw.empty:
         # Localize if naive, convert if aware
         if df_raw.index.tz is None:
             df_raw.index = df_raw.index.tz_localize('UTC')
         else:
             df_raw.index = df_raw.index.tz_convert('UTC')

    # Concatenate
    df_final = pd.concat([df_raw, df_streamed])

    # Remove duplicates (keep the latest version if overlaps exist)
    df_final = df_final[~df_final.index.duplicated(keep='last')]

    # Sort by time
    df_final.sort_index(inplace=True)

    # Save
    df_final.to_csv(output_path)
    print(f"Merged {len(df_raw)} historical records with {len(df_streamed)} streamed records.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()