import os
import joblib
import matplotlib.pyplot as plt
import sklearn.metrics as sklm

from sklearn.model_selection import train_test_split

from src.EstimDataLoader import EstimDataLoader
from src.EstimPreprocessor import EstimPreprocessor
from model.SpotifyPopularityModel import SpotifyPopularityModel



WK_DCT = os.getcwd()
MODEL_PATH = f"{WK_DCT}/estimer_popularite/model/spotify_model.pkl"

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = EstimPreprocessor()
X_train_proc = preprocessor.transform(X_train, text_columns)
X_test_proc = preprocessor.transform(X_test, text_columns)

model = SpotifyPopularityModel()
model.train(X_train_proc, y_train)


y_pred = model.predict(X_test_proc)


model.save(path = MODEL_PATH)


print("Model bien sauvegarder")
