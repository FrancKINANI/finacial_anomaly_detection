"""
Module pour la fusion de datasets financiers.
"""

import pandas as pd
import numpy as np

def merge_datasets(df_list, join_type='inner', on=None):
    """
    Fusionne une liste de DataFrames.
    """
    if not df_list:
        return None
    
    if len(df_list) == 1:
        return df_list[0]
        
    result = df_list[0]
    for df in df_list[1:]:
        if on:
            result = pd.merge(result, df, on=on, how=join_type)
        else:
            # Si pas de colonne commune spécifiée, on essaie une concaténation
            # ou on laisse pandas décider (pas recommandé pour données temporelles/entités)
            result = pd.concat([result, df], axis=0, ignore_index=True)
            
    return result

def check_alignment(df1, df2):
    """
    Vérifie si deux datasets ont des colonnes en commun.
    """
    common = set(df1.columns).intersection(set(df2.columns))
    return list(common)
