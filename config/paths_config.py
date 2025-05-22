import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw Data
RAW_DIR = os.path.join(BASE_DIR, "../artifacts/raw")

# Processed Data
PROCESSED_DIR = os.path.join(BASE_DIR, "../artifacts/processed")

# Model
MODEL_DIR = os.path.join(BASE_DIR, "../artifacts/model")
