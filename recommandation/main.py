from src.preprocessing.cleaner import SpotifyDataCleaner
from src.recommender.content_based import ContentBasedRecommender
import random

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

    mean_similarity = recommender.evaluate_similarity(example_id, top_k=5)
    print(f"\nMean similarity score: {mean_similarity:.3f}")

def evaluate_model(recommender, sample_size=20):
    sample_ids = random.sample(
        list(recommender.df["spotify_id"]),
        sample_size
    )

    scores = [
        recommender.evaluate_similarity(song_id)
        for song_id in sample_ids
    ]

    average_score = sum(scores) / len(scores)

    print(f"\nAverage similarity over {sample_size} songs: {average_score:.3f}")

if __name__ == "__main__":
    preprocess()
    run_recommender()

    # Évaluation
    recommender = ContentBasedRecommender(
        data_path=PROCESSED_PATH,
        features=FEATURES
    )
    recommender.load_data()
    recommender.fit()

    evaluate_model(recommender, sample_size=20)