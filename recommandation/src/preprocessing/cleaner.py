import pandas as pd
from sklearn.preprocessing import StandardScaler


class SpotifyDataCleaner:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.input_path)

    def clean(self):
        self.df.drop_duplicates(subset="spotify_id", inplace=True)
        self.df.dropna(inplace=True)

    def normalize(self, features: list):
        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

    def save(self, output_path: str):
        self.df.to_csv(output_path, index=False)
