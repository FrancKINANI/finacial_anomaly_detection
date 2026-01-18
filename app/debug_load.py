import pickle
from pathlib import Path
import os
import sys
import numpy as np
import sklearn

print(f"Numpy version: {np.__version__}")
print(f"Sklearn version: {sklearn.__version__}")

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'app' else SCRIPT_DIR
MODEL_PATH = BASE_DIR / 'models' / 'best_model.pkl'

print(f"Checking model at: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

FEATURE_PATH = BASE_DIR / 'models' / 'feature_names.pkl'
try:
    with open(FEATURE_PATH, 'rb') as f:
        features = pickle.load(f)
    print("Features loaded successfully!")
    print(f"Number of features: {len(features)}")
except Exception as e:
    print(f"Error loading features: {e}")
