from src.preprocessing.cleaner import SpotifyDataCleaner
from src.recommender.content_based import ContentBasedRecommender

FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

RAW_PATH = "data/raw/universal_top_spotify_songs.csv"
PROCESSED_PATH = "data/processed/spotify_clean.csv"


def preprocess():
    cleaner = SpotifyDataCleaner(input_path=RAW_PATH)
    cleaner.load()
    cleaner.clean()
    cleaner.normalize(FEATURES)
    cleaner.save(PROCESSED_PATH)


def run_recommender():
    recommender = ContentBasedRecommender(
        data_path=PROCESSED_PATH,
        features=FEATURES
    )

    recommender.load_data()
    recommender.fit()

    # Exemple : première musique du dataset
    example_id = recommender.df.iloc[0]["spotify_id"]

    recommendations = recommender.recommend(example_id, top_k=5)

    print("\nRecommended songs:\n")
    print(recommendations)

    print("Original song:")
    print(recommender.df[recommender.df["spotify_id"] == example_id][["name", "artists"]])


if __name__ == "__main__":
    preprocess()
    run_recommender()