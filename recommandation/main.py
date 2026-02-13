from src.preprocessing.cleaner import SpotifyDataCleaner

FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

cleaner = SpotifyDataCleaner(
    input_path="data/raw/universal_top_spotify_songs.csv"
)

cleaner.load()
cleaner.clean()
cleaner.normalize(FEATURES)
cleaner.save("data/processed/spotify_clean.csv")
