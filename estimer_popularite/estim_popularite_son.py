import os
import matplotlib.pyplot as plt
import sklearn.metrics as sklm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import src.EstimPipeline as ep
from src.EstimDataLoader import EstimDataLoader
from src.EstimPreprocessor import EstimPreprocessor
from model.SpotifyPopularityModel import SpotifyPopularityModel


WK_DCT = os.getcwd()

dataloader = EstimDataLoader(f"{WK_DCT}/data/spotify_top_song_day.csv")
top_day , text_columns = dataloader.load()

top_day = top_day.drop(columns=["spotify_id"])

text_columns.remove("spotify_id")

print("Text columns:", text_columns)

y_day = top_day["daily_rank"]
x_day = top_day.drop(columns=["daily_rank"])


X_train, X_test, y_train, y_test = train_test_split(
    x_day, y_day, test_size=0.2, random_state=42
)


preprocessor = EstimPreprocessor()
X_train_processed = preprocessor.transform(X_train, text_columns)
X_test_processed = preprocessor.transform(X_test,text_columns)


model = SpotifyPopularityModel()
model.train(X_train_processed, y_train)


y_pred = model.predict(X_test_processed)


rmse = sklm.root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")



