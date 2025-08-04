# preprocess_data.py
import os
import pandas as pd

# This script must be run from the root of the 'python-api' directory.
# Make sure you have run 'pip install pyarrow' first.

# Import the data loading function from your main API script
from predict_location_api import load_and_process_geojson

print("🚀 Starting one-time data pre-processing...")

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'ml')
output_path = os.path.join(data_path, 'processed_data')

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# --- Define the South Zone files we need to process ---
files_to_process = {
    'south_places_and_pois': [
        os.path.join(data_path, 'south_points_places.geojson'),
        os.path.join(data_path, 'north_pois.geojson'),
        os.path.join(data_path, 'west_pois.geojson'),
        os.path.join(data_path, 'center_pois.geojson')
    ],
    'south_waters': [
        os.path.join(data_path, 'south_points_waters.geojson')
    ],
    'south_landuse': [
        os.path.join(data_path, 'south_landuse.geojson')
    ]
}

# --- Process and save each dataset as a Feather file ---
for name, file_list in files_to_process.items():
    print(f"Processing {name}...")
    
    # Load and combine the geojson files into a single DataFrame
    df_list = [load_and_process_geojson(f) for f in file_list if os.path.exists(f)]
    if not df_list:
        print(f"⚠️  No files found for {name}, skipping.")
        continue

    df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)
    
    # Define the output file path
    output_file = os.path.join(output_path, f"{name}.feather")
    
    # Save the DataFrame to the highly efficient Feather format
    df.to_feather(output_file)
    print(f"✅ Saved {len(df)} rows to {output_file}")

print("\n🎉 Pre-processing complete!")