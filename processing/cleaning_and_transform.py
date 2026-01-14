"""
Module de prétraitement automatique des données financières
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

class FinancialPreprocessor:
    """Classe pour le prétraitement automatique"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_cols = None
        self.target_col = None
    
    def fit_transform(self, df, target_col=None):
        """
        Prétraite automatiquement un dataset
        
        Args:
            df (pd.DataFrame): Dataset brut
            target_col (str): Nom de la colonne cible (optionnel)
        
        Returns:
            pd.DataFrame: Dataset nettoyé
        """
        df_clean = df.copy()
        
        # Détecter la cible
        if target_col is None:
            target_col = self._auto_detect_target(df_clean)
        self.target_col = target_col
        
        # Identifier colonnes numériques
        self.numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
        
        # Nettoyage
        df_clean = self._handle_missing(df_clean)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._normalize(df_clean)
        
        return df_clean
    
    def _auto_detect_target(self, df):
        """Détecte automatiquement la variable cible"""
        keywords = ['bankrupt', 'default', 'failure', 'class', 'target', 'label']
        
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                if df[col].nunique() <= 5:
                    return col
        return None
    
    def _handle_missing(self, df):
        """Traite les valeurs manquantes"""
        # Supprimer colonnes avec >50% manquantes
        threshold = 0.5
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
        df = df[cols_to_keep]
        
        # Imputer le reste
        if len(self.numeric_cols) > 0:
            valid_numeric = [c for c in self.numeric_cols if c in df.columns]
            df[valid_numeric] = self.imputer.fit_transform(df[valid_numeric])
        
        return df
    
    def _handle_outliers(self, df):
        """Traite les outliers avec winsorization"""
        for col in self.numeric_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _normalize(self, df):
        """Normalise les données"""
        valid_numeric = [c for c in self.numeric_cols if c in df.columns]
        if len(valid_numeric) > 0:
            df[valid_numeric] = self.scaler.fit_transform(df[valid_numeric])
        
        return df
    
    def get_summary(self, df_original, df_clean):
        """Génère un résumé du prétraitement"""
        summary = {
            'original_shape': df_original.shape,
            'clean_shape': df_clean.shape,
            'rows_removed': df_original.shape[0] - df_clean.shape[0],
            'cols_removed': df_original.shape[1] - df_clean.shape[1],
            'missing_before': df_original.isnull().sum().sum(),
            'missing_after': df_clean.isnull().sum().sum(),
            'target_detected': self.target_col
        }
        return summary

def quick_clean(df):
    """
    Nettoyage rapide d'un dataset
    
    Args:
        df (pd.DataFrame): Dataset brut
    
    Returns:
        pd.DataFrame: Dataset nettoyé
    """
    preprocessor = FinancialPreprocessor()
    return preprocessor.fit_transform(df)