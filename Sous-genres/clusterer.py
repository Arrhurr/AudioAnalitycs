"""
Classe pour effectuer le clustering et l'analyse
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter


class MusicClusterer:
    """Classe pour le clustering de musiques"""
    
    def __init__(self, df, features):
        """
        Initialise le clusterer
        
        Args:
            df: DataFrame avec les données
            features: liste des features audio à utiliser
        """
        self.df = df.copy()
        self.features = features
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.model = None
        self.labels = None
    
    def normalize(self):
        """Normalise les features (moyenne=0, std=1)"""
        print(" Normalisation des features...")
        
        X = self.df[self.features]
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f" Features normalisées : {self.X_scaled.shape}\n")
        return self
    
    def find_best_k(self, k_range=[3, 5, 8, 10]):
        """
        Teste différentes valeurs de K pour trouver la meilleure
        
        Args:
            k_range: liste des K à tester
            
        Returns:
            meilleur K
        """
        
        print("RECHERCHE DU MEILLEUR NOMBRE DE CLUSTERS")
    
        
        best_k = k_range[0]
        best_score = -1
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            silhouette = silhouette_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            
            print(f"K={k:2d} → Silhouette: {silhouette:.3f} | Davies-Bouldin: {davies_bouldin:.3f}")
            
            if silhouette > best_score:
                best_score = silhouette
                best_k = k
        
        print(f"\n Meilleur K : {best_k} (Silhouette: {best_score:.3f})\n")
        return best_k
    
    def apply_clustering(self, n_clusters):
        """
        Applique le clustering K-Means
        
        Args:
            n_clusters: nombre de clusters
        """
        print(f" Application du clustering avec K={n_clusters}...")
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(self.X_scaled)
        self.df['cluster'] = self.labels
        
        print(f" Clustering terminé\n")
        self.show_cluster_distribution()
        return self
    
    def show_cluster_distribution(self):
        """Affiche la distribution des clusters"""
        print(" Distribution des clusters :")
        
        for cluster_id in sorted(np.unique(self.labels)):
            count = (self.labels == cluster_id).sum()
            pct = (count / len(self.labels)) * 100
            print(f"  Cluster {cluster_id} : {count:6,} chansons ({pct:5.2f}%)")
        
        print()
    
    def analyze_clusters(self):
        """Analyse les profils moyens de chaque cluster"""
       
        print("PROFILS DES CLUSTERS (Moyennes des features)")
      
        
        profiles = self.df.groupby('cluster')[self.features].mean().round(3)
        print("\n" + profiles.to_string() + "\n")
        
        return profiles
    
    def show_top_genres_per_cluster(self, top_n=5):
        """
        Affiche les genres dominants dans chaque cluster
        
        Args:
            top_n: nombre de genres à afficher par cluster
        """
        print("=" * 60)
        print(f"TOP {top_n} GENRES PAR CLUSTER")
        print("=" * 60)
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_songs = self.df[self.df['cluster'] == cluster_id]
            
            # Extraire tous les genres du cluster
            all_genres = []
            for genres_list in cluster_songs['genres_list']:
                if isinstance(genres_list, list):
                    all_genres.extend(genres_list)
            
            # Top N genres
            genre_counts = Counter(all_genres).most_common(top_n)
            
            print(f"\n Cluster {cluster_id} ({len(cluster_songs):,} chansons) :")
            for i, (genre, count) in enumerate(genre_counts, 1):
                pct = (count / len(cluster_songs)) * 100
                print(f"  {i}. {genre:35s} : {count:5,} ({pct:5.1f}%)")
        
        print()
    
    def show_popular_songs_per_cluster(self, n=5):
        """
        Affiche les chansons les plus populaires de chaque cluster
        
        Args:
            n: nombre de chansons par cluster
        """
        print("=" * 60)
        print(f"TOP {n} CHANSONS POPULAIRES PAR CLUSTER")
        print("=" * 60)
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_songs = self.df[self.df['cluster'] == cluster_id]
            
            print(f"\n Cluster {cluster_id} :")
            print("-" * 80)
            
            top_songs = cluster_songs.nlargest(n, 'popularity')[['name', 'artists', 'popularity']]
            
            for idx, row in top_songs.iterrows():
                # Parser les artistes
                try:
                    import ast
                    artists = ast.literal_eval(row['artists']) if isinstance(row['artists'], str) else row['artists']
                    artist_str = artists[0] if isinstance(artists, list) and len(artists) > 0 else "Unknown"
                except:
                    artist_str = str(row['artists'])[:25]
                
                print(f"  {row['popularity']:2.0f} | {row['name'][:45]:45s} - {artist_str[:25]:25s}")
        
        print()
    
    def get_results(self):
        """Retourne le DataFrame avec les clusters"""
        return self.df