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
        # Streamed data might not have a header if created incrementally, 
        # but your bot code does write headers for new files. 
        # We assume standard CSV format here matching raw data.
        try:
            df_streamed = pd.read_csv(streamed_path, parse_dates=['time'], index_col='time')
        except ValueError:
            # If dates fail to parse, specific handling might be needed based on bot output format
            df_streamed = pd.read_csv(streamed_path)
            if 'time' in df_streamed.columns:
                df_streamed['time'] = pd.to_datetime(df_streamed['time'])
                df_streamed.set_index('time', inplace=True)
    else:
        df_streamed = pd.DataFrame()

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