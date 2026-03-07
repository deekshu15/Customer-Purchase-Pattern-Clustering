import streamlit as st
import pandas as pd

from config import *
from clustering import *
from utils import *

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Studio",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# PAGE STATE
# ---------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "overview"

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* ----- Centered Title ----- */

.main-title{
text-align:center;
font-size:48px;
font-weight:700;
margin-top:10px;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#9ca3af;
margin-bottom:35px;
}

/* ----- Navigation Buttons ----- */

.stButton > button{
background:#111827;
border:1px solid #374151;
padding:12px 28px;
border-radius:10px;
font-weight:600;
transition:all 0.25s ease;
width:100%;
}

.stButton > button:hover{
background:#2f5d50;
color:white;
transform:translateY(-2px);
box-shadow:0 6px 12px rgba(0,0,0,0.25);
}

/* ----- Feature Card ----- */

.feature-card{
background:#1f2937;
padding:20px;
border-radius:12px;
border:1px solid #374151;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_INPUT_PATH)

df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Dashboard Controls")

st.sidebar.subheader("Cluster Parameters")

min_k = st.sidebar.slider("Min clusters", 2, 10, MIN_CLUSTERS)
max_k = st.sidebar.slider("Max clusters", 3, 15, MAX_CLUSTERS)

random_state = st.sidebar.number_input(
    "Random Seed",
    value=RANDOM_STATE
)

mode = st.sidebar.radio(
    "Cluster Selection Mode",
    ["Auto", "Manual"]
)

# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------

model_df, scaled_df = preprocess_data(
    df,
    FEATURE_COLUMNS
)

k_values, wcss, sil_scores = evaluate_clusters(
    scaled_df,
    min_k,
    max_k,
    random_state
)

best_k = k_values[sil_scores.index(max(sil_scores))]

if mode == "Manual":
    selected_k = st.sidebar.slider(
        "Select K",
        min_k,
        max_k,
        best_k
    )
else:
    selected_k = best_k

labels = run_kmeans(
    scaled_df,
    selected_k,
    random_state
)

model_df["Cluster"] = labels

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
'<div class="main-title">Customer Segmentation Studio</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Interactive machine learning dashboard for exploring customer purchasing behavior.</div>',
unsafe_allow_html=True
)

# ---------------------------------------------------
# NAVIGATION BUTTONS
# ---------------------------------------------------

b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("📊 Overview"):
        st.session_state.page = "overview"

with b2:
    if st.button("📉 Cluster Optimization"):
        st.session_state.page = "optimization"

with b3:
    if st.button("🧠 Customer Segments"):
        st.session_state.page = "segments"

with b4:
    if st.button("📁 Results"):
        st.session_state.page = "results"

st.markdown("---")

# ---------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------

if st.session_state.page == "overview":

    st.header("Dataset Overview")

    col1, col2 = st.columns([2,1])

    with col1:

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    with col2:

        st.markdown(
        """
        <div class="feature-card">

        <h4>Features used for clustering</h4>

        • Age  
        • Annual Income  
        • Spending Score

        </div>
        """,
        unsafe_allow_html=True
        )

# ---------------------------------------------------
# PAGE: CLUSTER OPTIMIZATION
# ---------------------------------------------------

elif st.session_state.page == "optimization":

    st.header("Finding Optimal Clusters")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            elbow_chart(
                k_values,
                wcss
            )
        )

    with col2:
        st.pyplot(
            silhouette_chart(
                k_values,
                sil_scores
            )
        )

    st.success(f"Best cluster count based on silhouette score: {best_k}")

# ---------------------------------------------------
# PAGE: CUSTOMER SEGMENTS
# ---------------------------------------------------

elif st.session_state.page == "segments":

    st.header("Customer Segments")

    st.pyplot(
        cluster_scatter(model_df),
        use_container_width=True
    )

    st.subheader("Cluster Profiles")

    profile = model_df.groupby(
        "Cluster"
    )[FEATURE_COLUMNS].mean().round(2)

    st.dataframe(profile, use_container_width=True)

# ---------------------------------------------------
# PAGE: RESULTS
# ---------------------------------------------------

elif st.session_state.page == "results":

    st.header("Export Results")

    st.dataframe(
        model_df,
        use_container_width=True
    )

    st.download_button(
        "Download CSV",
        data=to_csv_bytes(model_df),
        file_name="clustered_customers.csv"
    )