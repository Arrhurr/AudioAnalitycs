import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import estimer_popularite.src.EstimDataLoader as edl
import estimer_popularite.model.SpotifyPopularityModel as spm
import estimer_popularite.src.EstimPreprocessor as ep

# ==============================
# Estimer_popularite dataload and treatement before plot
# ==============================
WK_DCT = os.getcwd()
MODEL_PATH = f"{WK_DCT}/estimer_popularite/model/spotify_model.pkl"
PREPROCESSOR_PATH = f"{WK_DCT}/estimer_popularite/model/preprocessor.pkl"
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
X_proc = preprocessor.transform(X,text_columns)
y_pred = model.predict(X_proc)
# ==============================

# ==============================
# Function to plot Real vs Predicted
# ==============================
def plot_real_vs_pred(y_true, y_pred, title="Real vs Predicted Popularity"):
    """
    Scatter plot of real vs predicted values with:
    - Red line: linear trend of model
    """
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")

    # Linear trend
    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    plt.plot(x_line, poly1d_fn(x_line), color="red", linewidth=2, label="Linear trend")


    plt.xlabel("Real Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Basic scatter + trend
    plot_real_vs_pred(y, y_pred)

