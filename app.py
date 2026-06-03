import argparse

import pandas as pd

from config import *
from clustering import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer segmentation terminal application"
    )

    parser.add_argument(
        "--min-k",
        type=int,
        default=MIN_CLUSTERS,
        help="Minimum number of clusters to evaluate",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=MAX_CLUSTERS,
        help="Maximum number of clusters to evaluate",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible clustering",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="Choose automatic cluster selection or manual k selection",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Cluster count to use in manual mode",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DATA_OUTPUT_PATH,
        help="Output CSV path for clustered results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(DATA_INPUT_PATH)
    print("Loaded dataset:", DATA_INPUT_PATH)
    print("Rows:", len(df), "Columns:", len(df.columns))
    print("Features used:", ", ".join(FEATURE_COLUMNS))
    print()

    model_df, scaled_df = preprocess_data(df, FEATURE_COLUMNS)

    print(
        f"Evaluating clusters for k in [{args.min_k}, {args.max_k}] with random state {args.random_state}..."
    )

    k_values, wcss, sil_scores = evaluate_clusters(
        scaled_df,
        args.min_k,
        args.max_k,
        args.random_state,
    )

    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f"Best cluster count by silhouette score: {best_k}")
    print()

    if args.mode == "manual":
        if args.k is None:
            raise ValueError(
                "Manual mode requires --k to be provided."
            )

        if args.k < args.min_k or args.k > args.max_k:
            raise ValueError(
                "Selected k must be between min_k and max_k."
            )

        selected_k = args.k
        print(f"Manual mode selected: using k = {selected_k}")
    else:
        selected_k = best_k
        print("Auto mode selected: using best silhouette cluster count.")

    print()

    labels = run_kmeans(
        scaled_df,
        selected_k,
        args.random_state,
    )

    model_df["Cluster"] = labels

    counts = model_df["Cluster"].value_counts().sort_index()
    print("Cluster sizes:")
    for cluster_label, count in counts.items():
        print(f"  Cluster {cluster_label}: {count} rows")

    print()
    print("Silhouette scores by k:")
    for k, score in zip(k_values, sil_scores):
        print(f"  k={k}: {score:.4f}")

    print()
    print("Cluster profile means:")
    print(
        model_df.groupby("Cluster")[FEATURE_COLUMNS]
        .mean()
        .round(2)
        .to_string()
    )

    model_df.to_csv(args.output, index=False)
    print()
    print(f"Clustered dataset exported to: {args.output}")


if __name__ == "__main__":
    main()
