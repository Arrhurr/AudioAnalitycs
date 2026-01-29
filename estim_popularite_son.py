import os
import clean_data_estim as cde
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split



WK_DCT = os.getcwd()

top_week = cde.clean_data_estim(f"{WK_DCT}\data\spotify_top_song_week.csv")

y_week = top_week['daily_rank']
x_week = top_week.drop(columns=['daily_rank'])

x_week_train, x_week_test, y_week_train, y_week_test = train_test_split(
    x_week, y_week, test_size=0.2, random_state=42)
print(x_week_train.shape, x_week_test.shape)



