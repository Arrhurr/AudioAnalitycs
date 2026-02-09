"""
Classe pour créer les visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


class ClusterVisualizer:
    """Classe pour créer les visualisations des clusters"""
    
    def __init__(self, X_scaled, labels, output_prefix='results'):
        """
        Initialise le visualiseur
        
        Args:
            X_scaled: données normalisées
            labels: labels des clusters
            output_prefix: préfixe pour les fichiers de sortie
        """
        self.X_scaled = X_scaled
        self.labels = labels
        self.output_prefix = output_prefix
    
    def plot_tsne(self, sample_size=5000):
        """
        Crée une visualisation t-SNE 2D
        
        Args:
            sample_size: nombre de points à afficher
        """
        print(" Génération de la visualisation t-SNE...")
        
        # Échantillonner si nécessaire
        if len(self.X_scaled) > sample_size:
            print(f"   (Échantillonnage de {sample_size:,} points)")
            indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[indices]
            labels_sample = self.labels[indices]
        else:
            X_sample = self.X_scaled
            labels_sample = self.labels
        
        # Calculer t-SNE
        print("   Calcul de t-SNE (1-2 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=0)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        n_clusters = len(np.unique(labels_sample))
        colors = sns.color_palette("husl", n_clusters)
        
        for cluster_id in sorted(np.unique(labels_sample)):
            mask = labels_sample == cluster_id
            ax.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[colors[cluster_id]],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title('Visualisation t-SNE des Clusters', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)
        
        output_file = f'{self.output_prefix}_tsne.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f" Sauvegardé : {output_file}\n")
        plt.close()
    
    def plot_feature_distributions(self, df, features):
        """
        Affiche les distributions des features
        
        Args:
            df: DataFrame
            features: liste des features
        """
        print(" Génération des distributions des features...")
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            df[feature].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_title(feature, fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Valeur', fontsize=9)
            axes[idx].set_ylabel('Fréquence', fontsize=9)
            axes[idx].grid(alpha=0.3)
        
        # Cacher les axes vides
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        output_file = f'{self.output_prefix}_distributions.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f" Sauvegardé : {output_file}\n")
        plt.close()
    
    def plot_correlation_matrix(self, df, features):
        """
        Affiche la matrice de corrélation
        
        Args:
            df: DataFrame
            features: liste des features
        """
        print(" Génération de la matrice de corrélation...")
        
        corr = df[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation - Features Audio', fontsize=14, fontweight='bold')
        
        output_file = f'{self.output_prefix}_correlation.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f" Sauvegardé : {output_file}\n")
        plt.close()