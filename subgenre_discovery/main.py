"""
Script principal pour l'analyse de clustering musical

Usage: python main.py
"""

from data_processor import DataProcessor
from clusterer import MusicClusterer
from visualizer import ClusterVisualizer

# ============================================================================
# CONFIGURATION - Modifier ces valeurs selon tes besoins
# ============================================================================

DATA_FILE = "C:/Users/merch/Documents/Projet_python_avancé/Dataset_with_Genres.csv"  # Chemin vers le dataset
GENRE = 'rock'                                    # Genre à analyser
N_CLUSTERS = 5                                    # Nombre de clusters (ou None pour auto)

# Features audio à utiliser pour le clustering
AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

def main():
    """Fonction principale"""
    
    print("=" * 60)
    print("AUDIOANALYTICS - DÉCOUVERTE DE SOUS-GENRES")
    print("=" * 60)
    print()
    
    
    # ÉTAPE 1 : Charger et préparer les données
    
    
    processor = DataProcessor(DATA_FILE)
    processor.load_data()
    processor.convert_genres()
    processor.show_top_genres(top_n=20)
    processor.filter_by_genre(GENRE)
    processor.clean_data()
    processor.show_popular_songs(n=10)
    
    df = processor.get_data()
    
  
    # ÉTAPE 2 : Clustering
    
    
    clusterer = MusicClusterer(df, AUDIO_FEATURES)
    clusterer.normalize()
    
    # Trouver le meilleur K si non spécifié
    if N_CLUSTERS is None:
        best_k = clusterer.find_best_k(k_range=[3, 5, 8, 10, 12])
        clusterer.apply_clustering(best_k)
    else:
        clusterer.apply_clustering(N_CLUSTERS)
    
    
    # ÉTAPE 3 : Analyse des clusters
    
    
    profiles = clusterer.analyze_clusters()
    clusterer.show_top_genres_per_cluster(top_n=5)
    clusterer.show_popular_songs_per_cluster(n=5)
    
    df_results = clusterer.get_results()
    
    
    # ÉTAPE 4 : Visualisations
    
    
    print("=" * 60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 60)
    print()
    
    visualizer = ClusterVisualizer(
        clusterer.X_scaled,
        clusterer.labels,
        output_prefix=GENRE
    )
    
    visualizer.plot_tsne(sample_size=5000)
    visualizer.plot_feature_distributions(df_results, AUDIO_FEATURES)
    visualizer.plot_correlation_matrix(df_results, AUDIO_FEATURES)
    
    
    # ÉTAPE 5 : Sauvegarder les résultats
    
    
    output_file = f'{GENRE}_clusters.csv'
    df_results.to_csv(output_file, index=False)
    print(f" Résultats sauvegardés : {output_file}\n")
    
    # ========================================
    # RÉSUMÉ FINAL
    # ========================================
    
    print("=" * 60)
    print(" ANALYSE TERMINÉE !")
    print("=" * 60)
    print()
    print(" Fichiers générés :")
    print(f"   • {output_file}")
    print(f"   • {GENRE}_tsne.png")
    print(f"   • {GENRE}_distributions.png")
    print(f"   • {GENRE}_correlation.png")
    print()
    print(f" {len(df_results):,} chansons analysées")
    print(f" {len(df_results['cluster'].unique())} clusters découverts")
    print()


if __name__ == "__main__":
    main()