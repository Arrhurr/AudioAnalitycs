import os
import clean_data_estim as cde
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as sklpr

import sklearn.metrics as sklm
# Calcule l'erreur quadratique moyenne (MSE) entre les étiquettes réelles et les prédictions du modèle linéaire



ordinal = sklpr.OrdinalEncoder() # Initialise un encodeur ordinal pour convertir les catégories textuelles en nombres


WK_DCT = os.getcwd()

top_day = cde.clean_data_estim(f"{WK_DCT}\data\spotify_top_song_day.csv")
top_day = top_day.drop(columns=["spotify_id"])


y_day = top_day['daily_rank']
x_day = top_day.drop(columns=['daily_rank'])

column_text = ["name","artists","country","snapshot_date","is_explicit","album_name","album_release_date"]

for text in column_text:
    train_cat = x_day[[text]]
    x_day[text] = ordinal.fit_transform(train_cat)


x_day_train, x_day_test, y_day_train, y_day_test = train_test_split(x_day, y_day, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_day_train, y_day_train)
y_day_pred = model.predict(x_day_test)

rmse = sklm.mean_squared_error(y_day_test,y_day_pred)
print(rmse**(1/2)) # Calcule et affiche la racine carrée de l'erreur quadratique moyenne (RMSE)




