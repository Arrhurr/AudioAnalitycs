import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based music recommender using cosine similarity
    on Spotify audio features.
    """

    def __init__(self, data_path: str, features: list):
        self.data_path = data_path
        self.features = features
        self.df = None
        self.similarity_matrix = None

    def load_data(self):
        """Load preprocessed dataset."""
        self.df = pd.read_csv(self.data_path)

    def fit(self):
        """
        Compute cosine similarity matrix between songs.
        """
        feature_matrix = self.df[self.features]
        self.similarity_matrix = cosine_similarity(feature_matrix)

    def recommend(self, song_id: str, top_k: int = 5):
        """
        Recommend top_k similar songs to the given song_id.
        Returns recommendations with similarity scores.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        song_index = self.df.index[self.df["spotify_id"] == song_id].tolist()

        if not song_index:
            raise ValueError("Song ID not found in dataset.")

        song_index = song_index[0]

        similarity_scores = list(enumerate(self.similarity_matrix[song_index]))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        similarity_scores = similarity_scores[1: top_k + 1]

        recommended_indices = [i[0] for i in similarity_scores]
        scores = [i[1] for i in similarity_scores]

        recommendations = self.df.iloc[recommended_indices][
            ["spotify_id", "name", "artists"]
        ].copy()

        recommendations["similarity_score"] = scores

        return recommendations
    
    def evaluate_similarity(self, song_id: str, top_k: int = 5):
        recommendations = self.recommend(song_id, top_k)
        return recommendations["similarity_score"].mean()