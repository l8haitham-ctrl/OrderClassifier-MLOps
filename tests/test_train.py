import os
import sys
import pandas as pd
import pytest

# Add the parent directory to the path so we can import from train.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_and_prepare_data, DATA_PATH

def test_load_and_prepare_data():
    """Unit test for the data preparation function."""
    # Ensure the data file exists
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    
    # Call the function
    X, y, features = load_and_prepare_data(DATA_PATH)
    
    # Check return types
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert len(y) == len(X), "X and y must have the same number of rows"
    
    # Check expected features
    expected_features = [
        "amount", "customer_id", "product_category_enc", "region_enc",
        "day_of_week", "month", "quarter", "is_weekend"
    ]
    for feat in expected_features:
        assert feat in features, f"Feature {feat} is missing from feature list"
        assert feat in X.columns, f"Feature {feat} is missing from X"

    # Check for nulls
    assert X.isnull().sum().sum() == 0, "X should not contain any null values"
