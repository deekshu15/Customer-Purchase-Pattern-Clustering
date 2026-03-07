import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def elbow_chart(k_values, wcss):

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(k_values, wcss, marker="o")

    ax.set_title("Elbow Curve")

    ax.set_xlabel("Number of Clusters")

    ax.set_ylabel("WCSS")

    ax.grid(alpha=0.3)

    return fig


def silhouette_chart(k_values, scores):

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(k_values, scores, marker="o")

    ax.set_title("Silhouette Score")

    ax.set_xlabel("Number of Clusters")

    ax.set_ylabel("Score")

    ax.grid(alpha=0.3)

    return fig


def cluster_scatter(df):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set2",
        s=90,
        ax=ax
    )

    ax.set_title(
        "Customer Segments"
    )

    return fig


def to_csv_bytes(df):

    return df.to_csv(index=False).encode("utf-8")