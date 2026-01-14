"""
Module de sélection de caractéristiques.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold

class FeatureSelector:
    """Classe pour la sélection automatique de caractéristiques"""
    
    def __init__(self, target_col):
        self.target_col = target_col
        self.selected_features = None
        self.scores_ = None
    
    def remove_low_variance(self, df, threshold=0.01):
        """Supprime les variables avec une variance trop faible"""
        X = df.drop(columns=[self.target_col])
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return df[[self.target_col] + X.columns[selector.get_support()].tolist()]
    
    def select_k_best(self, df, k=20, method=f_classif):
        """Sélectionne les k meilleures variables selon une métrique"""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        selector = SelectKBest(score_func=method, k=k)
        selector.fit(X, y)
        
        cols = X.columns[selector.get_support()].tolist()
        self.selected_features = cols
        self.scores_ = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values(by='Score', ascending=False)
        
        return df[[self.target_col] + cols]

    def select_by_mutual_info(self, df, k=20):
        """Sélectionne par information mutuelle"""
        return self.select_k_best(df, k=k, method=mutual_info_classif)
