"""
Configuration file for Customer Segmentation Dashboard
"""

# Data paths
DATA_INPUT_PATH = "data/Mall_Customers.csv"
DATA_OUTPUT_PATH = "data/clustered_customers.csv"

# Clustering settings
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
RANDOM_STATE = 42

# Features used
FEATURE_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]