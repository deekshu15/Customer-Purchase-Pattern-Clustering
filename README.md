# Customer Purchase Pattern Clustering

A machine learning project that performs customer segmentation using K-Means clustering on mall customer data.

## Project Overview

This project analyzes customer behavior and segments them into distinct groups based on:
- **Age**
- **Annual Income**
- **Spending Score**

The analysis helps businesses understand customer patterns and tailor marketing strategies accordingly.

## Dataset

- **File**: `data/Mall_Customers.csv`
- **Features**: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- **Samples**: ~200 customer records

## Project Structure

```
├── mall_customer_segmentation.ipynb   # Main analysis notebook
├── clustering_analysis.py             # Standalone Python script
├── config.py                          # Configuration parameters
├── requirements.txt                   # Python dependencies
├── data/
│   ├── Mall_Customers.csv             # Input dataset
│   └── clustered_customers.csv        # Output with cluster assignments
└── README.md                          # This file
```

## Installation

1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the Jupyter Notebook
```bash
jupyter notebook mall_customer_segmentation.ipynb
```
Then execute all cells sequentially.

### Option 2: Run the Python Script
```bash
python clustering_analysis.py
```

### Option 3: Run the Streamlit App
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Streamlit Deployment

### Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in.
3. Click **New app** and select your repository/branch.
4. Set the main file path to `app.py`.
5. Deploy.

### Deployment Notes
- Make sure `requirements.txt` is present (already included).
- Keep `data/Mall_Customers.csv` in the repository so the app can read the dataset.
- If you change file paths, update `config.py`.

## Analysis Workflow

1. **Data Loading & Preprocessing**
   - Load customer data
   - Encode categorical variables (Gender)
   - Scale features using StandardScaler

2. **Optimal Cluster Finding**
   - Elbow Method (WCSS analysis)
   - Silhouette Score evaluation
   - Range: 2-10 clusters

3. **K-Means Clustering**
   - Apply optimal K-Means model
   - Assign customers to clusters

4. **Analysis & Visualization**
   - Cluster statistics (sizes, mean features)
   - Gender distribution per cluster
   - 2D pair plots
   - 3D scatter plot visualization
   - Silhouette score evaluation

5. **Results Export**
   - Save clustered data to `data/clustered_customers.csv`

## Key Outputs

- **Optimal Clusters**: Determined automatically (usually 3-5)
- **Cluster Profiles**: Average age, income, and spending score per cluster
- **Visualizations**: 
  - Elbow curve
  - Silhouette scores
  - 2D pair plots
  - 3D cluster visualization
- **Quality Metric**: Silhouette score (values closer to 1 indicate better clustering)

## Configuration

Edit `config.py` to customize:
- Cluster range (default: 2-10)
- Random state for reproducibility
- Input/output file paths
- Feature selection

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions

