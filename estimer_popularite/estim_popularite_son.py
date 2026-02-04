import os
import matplotlib.pyplot as plt
import sklearn.metrics as sklm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import src.EstimPipeline as ep
from src.EstimDataLoader import EstimDataLoader
from src.EstimPreprocessor import EstimPreprocessor
from model.SpotifyPopularityModel import SpotifyPopularityModel

from sklearn.metrics import accuracy_score


WK_DCT = os.getcwd()

dataloader = EstimDataLoader(f"{WK_DCT}/data/spotify_top_song_day.csv")
top_day , text_columns = dataloader.load()

drop = ["spotify_id","loudness","key","album_release_date","duration_ms","valence","energy","tempo","liveness","country","mode","album_name","is_explicit","snapshot_date","acousticness" ,"danceability","instrumentalness","artists"]
for col in drop:
    top_day = top_day.drop(columns=[col])
    if(col in text_columns):
        text_columns.remove(col)

preprocessor = EstimPreprocessor()

top_day_processed = preprocessor.transform(top_day, text_columns)

corr_matrix = top_day_processed.corr()
print(corr_matrix)
target_corr = corr_matrix["popularity"].sort_values(
    key=abs,
    ascending=False
)

print(target_corr)



y_day = top_day["popularity"]
x_day = top_day.drop(columns=["popularity"])


X_train_processed, X_test_processed, y_train, y_test = train_test_split(
    x_day, y_day, test_size=0.2, random_state=42
)







model = SpotifyPopularityModel()
model.train(X_train_processed, y_train)


y_pred = model.predict(X_test_processed)


rmse = sklm.root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real daily rank")
plt.ylabel("Predicted daily rank")
plt.title("Real vs Predicted Spotify Daily Rank")
#plt.plot(
#    [y_test.min(), y_test.max()],
#    [y_test.min(), y_test.max()],
#    "r--"
#)
#plt.show()



print(f"accuracy : {accuracy_score(y_test, y_pred.round())}")