"""
Classe pour traiter et préparer les données musicales
"""

import ast
import pandas as pd
from collections import Counter


class DataProcessor:
    """Classe pour charger et préparer les données"""
    
    def __init__(self, filepath):
        """
        Initialise le processeur de données
        
        Args:
            filepath: chemin vers le fichier CSV
        """
        self.filepath = filepath
        self.df = None
        self.df_filtered = None
        
    def load_data(self):
        """Charge le dataset"""
        print(f" Chargement du dataset '{self.filepath}'")
        self.df = pd.read_csv(self.filepath)
        
        return self
    
    def convert_genres(self, genre_column='consolidates_genre_lists'):
        """Convertit la colonne de genres en listes Python"""
        print(" Conversion des genres")
        
        def parse_genre(genre_str):
            if pd.isna(genre_str) or genre_str == '[]':
                return []
            try:
                return ast.literal_eval(genre_str)
            except:
                return []
        
        self.df['genres_list'] = self.df[genre_column].apply(parse_genre)
        print(f"✓ Genres convertis en listes\n")
        return self
    
    def show_top_genres(self, top_n=20):
        """Affiche les genres les plus populaires"""
        print(f"Top {top_n} genres disponibles :")
        
        all_genres = []
        for genres in self.df['genres_list']:
            if isinstance(genres, list):
                all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        
        for i, (genre, count) in enumerate(genre_counts.most_common(top_n), 1):
            pct = (count / len(self.df)) * 100
            print(f"  {i:2d}. {genre:35s} : {count:7,} chansons ({pct:5.2f}%)")
        
        print()
        return genre_counts
    
    def filter_by_genre(self, genre_keywords):
        """
        Filtre le dataset par genre(s)
        
        Args:
            genre_keywords: string ou liste (ex: 'rock' ou ['rap', 'hip'])
        """
        print(f" Filtrage par genre : '{genre_keywords}'...")
        
        if isinstance(genre_keywords, str):
            genre_keywords = [genre_keywords]
        
        mask = self.df['genres_list'].apply(
            lambda genres: any(
                any(keyword.lower() in g.lower() for keyword in genre_keywords)
                for g in genres
            ) if isinstance(genres, list) else False
        )
        
        self.df_filtered = self.df[mask].copy()
        print(f" {len(self.df_filtered):,} chansons sélectionnées\n")
        return self
    
    def clean_data(self):
        """Nettoie les données (supprime doublons)"""
        print("🧹 Nettoyage des données")
        
        if self.df_filtered is None:
            df_to_clean = self.df
        else:
            df_to_clean = self.df_filtered
        
        initial = len(df_to_clean)
        df_to_clean = df_to_clean.drop_duplicates(subset=['id'], keep='first')
        removed = initial - len(df_to_clean)
        
        print(f"✓ Chansons restantes : {len(df_to_clean):,}\n")
        
        if self.df_filtered is None:
            self.df = df_to_clean
        else:
            self.df_filtered = df_to_clean
        
        return self
    
    def get_data(self):
        """Retourne le DataFrame (filtré si disponible, sinon complet)"""
        return self.df_filtered if self.df_filtered is not None else self.df
    
    def show_popular_songs(self, n=10):
        """
        Affiche les chansons les plus populaires du dataset filtré
        
        Args:
            n: nombre de chansons à afficher
        """
        df = self.get_data()
        
        print(f"Top {n} chansons les plus populaires :")
        print("-" * 80)
        
        top_songs = df.nlargest(n, 'popularity')[['name', 'artists', 'popularity']]
        
        for idx, row in top_songs.iterrows():
            # Parser les artistes
            try:
                artists = ast.literal_eval(row['artists']) if isinstance(row['artists'], str) else row['artists']
                artist_str = artists[0] if isinstance(artists, list) and len(artists) > 0 else "Unknown"
            except:
                artist_str = str(row['artists'])[:30]
            
            print(f"  {row['popularity']:2.0f} | {row['name'][:50]:50s} - {artist_str[:30]:30s}")
        
        print()