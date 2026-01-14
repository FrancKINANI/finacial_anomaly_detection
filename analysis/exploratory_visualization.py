"""
Module de visualisation exploratoire des données financières.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_target_distribution(df, target_col, save_path=None):
    """
    Affiche et sauvegarde la distribution de la variable cible.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Distribution de la variable cible: {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Nombre d\'observations')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation_matrix(df, numeric_cols, save_path=None, top_n=20):
    """
    Affiche une matrice de corrélation pour les n variables les plus corrélées à la cible.
    """
    # Si on a trop de colonnes, on sélectionne les plus importantes
    if len(numeric_cols) > top_n:
        # Ici on pourrait filtrer, mais pour l'instant on prend juste les n premières
        cols = numeric_cols[:top_n]
    else:
        cols = numeric_cols
        
    plt.figure(figsize=(15, 12))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice de corrélation (échantillon de variables)')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_basic_stats(df):
    """
    Retourne des statistiques descriptives de base.
    """
    return df.describe()

def check_missing_values(df):
    """
    Affiche le nombre de valeurs manquantes par colonne.
    """
    missing = df.isnull().sum()
    return missing[missing > 0]
