import os
import joblib
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import model.SpotifyPopularityModel as spm
import src.EstimPreprocessor as ep

from src.EstimDataLoader import EstimDataLoader


WK_DCT = os.getcwd()
MODEL_PATH = f"{WK_DCT}/estimer_popularite/model/spotify_model.pkl"


model = spm.SpotifyPopularityModel()
model.load(path=MODEL_PATH)
preprocessor = ep.EstimPreprocessor()

dataloader = EstimDataLoader(
    f"{WK_DCT}/data/spotify_top_song_day.csv"
)
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

rmse = sklm.root_mean_squared_error(y, y_pred)
r2 = sklm.r2_score(y, y_pred)

print(f"RMSE (loaded model): {rmse:.2f}")
print(f"R² score (loaded model): {r2:.3f}")

# Visualization
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Real popularity")
plt.ylabel("Predicted popularity")
plt.title("Loaded model – Real vs Predicted")
plt.show()
