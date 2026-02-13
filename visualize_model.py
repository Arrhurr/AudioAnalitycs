import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

# ==============================
# PATH CONFIGURATION
# ==============================

WK_DCT = os.getcwd()

# Allow imports from recommendation project
sys.path.append(os.path.join(WK_DCT, "recommandation"))

# ==============================
# PART 1 — POPULARITY MODEL VISUALIZATION
# ==============================

import estimer_popularite.src.EstimDataLoader as edl
import estimer_popularite.model.SpotifyPopularityModel as spm
import estimer_popularite.src.EstimPreprocessor as ep

MODEL_PATH = f"{WK_DCT}/estimer_popularite/model/spotify_model.pkl"
DATA_PATH = f"{WK_DCT}/data/spotify_top_song_day.csv"

model = spm.SpotifyPopularityModel()
model.load(path=MODEL_PATH)

preprocessor = ep.EstimPreprocessor()
dataloader = edl.EstimDataLoader(DATA_PATH)

df, text_columns = dataloader.load()

drop = [
    "spotify_id", "loudness", "key", "album_release_date",
    "duration_ms", "valence", "energy", "tempo", "liveness",
    "country", "mode", "album_name", "is_explicit",
    "snapshot_date", "acousticness", "danceability",
    "instrumentalness", "artists"
]

df = df.drop(columns=drop)
text_columns = [c for c in text_columns if c not in drop]

y = df["popularity"]
X = df.drop(columns=["popularity"])
X_proc = preprocessor.transform(X, text_columns)
y_pred = model.predict(X_proc)


def plot_real_vs_pred(y_true, y_pred, title="Real vs Predicted Popularity"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    plt.plot(x_line, poly1d_fn(x_line), color="red", linewidth=2)

    plt.xlabel("Real Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title(title)
    plt.show()


# ==============================
# PART 2 — RECOMMENDATION VISUALIZATION
# ==============================

from recommandation.src.recommender.content_based import ContentBasedRecommender

REC_DATA_PATH = f"{WK_DCT}/recommandation/data/processed/spotify_clean.csv"

BASE_FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]


def plot_correlation_heatmap():
    df = pd.read_csv(REC_DATA_PATH)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[BASE_FEATURES].corr(), annot=True)
    plt.title("Audio Feature Correlation Heatmap")
    plt.show()


def evaluate_model(features, sample_size=20):
    recommender = ContentBasedRecommender(
        data_path=REC_DATA_PATH,
        features=features
    )
    recommender.load_data()
    recommender.fit()

    sample_ids = random.sample(
        list(recommender.df["spotify_id"]),
        sample_size
    )

    scores = [
        recommender.evaluate_similarity(song_id)
        for song_id in sample_ids
    ]

    return sum(scores) / len(scores)


def plot_feature_comparison():
    feature_sets = {
        "All features": BASE_FEATURES,
        "Without loudness": [f for f in BASE_FEATURES if f != "loudness"],
        "Without tempo": [f for f in BASE_FEATURES if f != "tempo"],
        "Without instrumentalness": [f for f in BASE_FEATURES if f != "instrumentalness"]
    }

    results = {}

    for name, features in feature_sets.items():
        score = evaluate_model(features)
        results[name] = score

    plt.figure(figsize=(8, 6))
    plt.bar(results.keys(), results.values())
    plt.xticks(rotation=45)
    plt.ylabel("Average Cosine Similarity")
    plt.title("Feature Impact on Recommendation Quality")
    plt.show()

# ==============================
# PART 3 — SUBGENRE DISCOVERY (CLUSTERING)
# ==============================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CLUSTER_DATA_PATH = f"{WK_DCT}/data/dataset_clean_rap.csv"

CLUSTER_FEATURES = [
    "danceability", "energy", "loudness", "tempo",
    "valence", "speechiness", "acousticness",
    "instrumentalness", "liveness"
]


def run_subgenre_clustering(k=4):
    df = pd.read_csv(CLUSTER_DATA_PATH)

    # Keep only available features
    features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[features].copy()

    # Convert to numeric
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.loc[X.index].copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(df["pca1"], df["pca2"], c=df["cluster"], alpha=0.6)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Subgenre Clusters (k={k})")
    plt.show()

    # Print cluster means
    cluster_means = df.groupby("cluster")[features].mean().round(3)
    print("\nCluster Means:\n")
    print(cluster_means)

    return df

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    # Popularity visualization
    plot_real_vs_pred(y, y_pred)

    # Recommendation visualizations
    plot_correlation_heatmap()
    plot_feature_comparison()
    run_subgenre_clustering(k=4)
