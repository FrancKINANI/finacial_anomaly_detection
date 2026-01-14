"""
Module de réduction de dimensionnalité.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class PCAReducer:
    """Classe pour la réduction de dimension via PCA"""
    
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        self.explained_variance_ = None
    
    def fit_transform(self, df, target_col):
        """Applique PCA sur les données"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_pca = self.pca.fit_transform(X)
        self.explained_variance_ = self.pca.explained_variance_ratio_
        
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols)
        df_pca[target_col] = y.values
        
        return df_pca
    
    def plot_explained_variance(self, save_path=None):
        """Affiche la variance expliquée cumulée"""
        if self.explained_variance_ is None:
            print("Erreur: PCA non fittée.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.explained_variance_), marker='o')
        plt.xlabel('Nombre de composants')
        plt.ylabel('Variance expliquée cumulée')
        plt.title('Analyse de la variance expliquée par PCA')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_2d_projection(self, df_pca, target_col, save_path=None):
        """Affiche une projection 2D des deux premiers composants"""
        if 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns:
            print("Erreur: Colonnes PC1/PC2 manquantes.")
            return

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue=target_col, data=df_pca, palette='viridis', alpha=0.6)
        plt.title('Projection PCA 2D')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
