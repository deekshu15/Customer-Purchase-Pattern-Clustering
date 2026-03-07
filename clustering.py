import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def preprocess_data(df, feature_columns):

    df_model = df.copy()

    if "CustomerID" in df_model.columns:
        df_model = df_model.drop("CustomerID", axis=1)

    if "Gender" in df_model.columns:
        df_model["Gender"] = df_model["Gender"].map(
            {"Male": 0, "Female": 1}
        )

    X = df_model[feature_columns]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(scaled, columns=feature_columns)

    return df_model, scaled_df


def evaluate_clusters(scaled_df, min_k, max_k, random_state):

    k_values = list(range(min_k, max_k + 1))

    wcss = []
    sil_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state)

        labels = model.fit_predict(scaled_df)

        wcss.append(model.inertia_)

        sil_scores.append(
            silhouette_score(scaled_df, labels)
        )

    return k_values, wcss, sil_scores


def run_kmeans(scaled_df, k, random_state):

    model = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    labels = model.fit_predict(scaled_df)

    return labels