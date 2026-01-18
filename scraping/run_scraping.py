"""
Module de Web Scraping pour récupérer des données financières
"""

import pandas as pd
import requests
from io import StringIO
import re
from .dynamic import playwright_scraper

def scrape_from_url(url, dynamic=False):
    """
    Scrape un dataset depuis une URL (Supporte statique et dynamique)
    
    Args:
        url (str): URL du fichier ou de la page
        dynamic (bool): Si True, utilise Playwright pour les pages dynamiques
    
    Returns:
        pd.DataFrame: Dataset chargé ou None si erreur
    """
    if dynamic:
        return playwright_scraper.scrape_dynamic_csv(url)
    
    try:
        # Télécharger le contenu
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Lire le CSV
        df = pd.read_csv(StringIO(response.text))
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion: {e}")
        return None
    
    except pd.errors.EmptyDataError:
        print("Le fichier est vide")
        return None
    
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return None

def scrape_github_csv(github_url):
    """
    Scrape un CSV depuis GitHub (convertit l'URL si nécessaire)
    
    Args:
        github_url (str): URL GitHub du fichier
    
    Returns:
        pd.DataFrame: Dataset ou None
    """
    # Convertir l'URL GitHub en raw URL
    if 'github.com' in github_url and 'raw' not in github_url:
        github_url = github_url.replace('github.com', 'raw.githubusercontent.com')
        github_url = github_url.replace('/blob/', '/')
    
    return scrape_from_url(github_url)

def scrape_kaggle_dataset(dataset_path):
    """
    Template pour scraper depuis Kaggle (nécessite authentification)
    
    Args:
        dataset_path (str): Chemin du dataset Kaggle (ex: 'username/dataset-name')
    
    Returns:
        str: Message d'instructions
    """
    instructions = f"""
    Pour télécharger depuis Kaggle:
    
    1. Installez: pip install kaggle
    2. Configurez votre API key (kaggle.json)
    3. Exécutez: kaggle datasets download -d {dataset_path}
    4. Uploadez le fichier téléchargé
    """
    return instructions

def validate_financial_data(df):
    """
    Valide qu'un dataset contient des données financières exploitables
    
    Args:
        df (pd.DataFrame): Dataset à valider
    
    Returns:
        tuple: (bool, str) - (est_valide, message)
    """
    if df is None or df.empty:
        return False, "Dataset vide"
    
    # Vérifier colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 5:
        return False, "Pas assez de colonnes numériques (minimum 5)"
    
    # Vérifier taille minimale
    if len(df) < 100:
        return False, "Dataset trop petit (minimum 100 lignes)"
    
    # Vérifier valeurs manquantes excessives
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 50:
        return False, f"Trop de valeurs manquantes ({missing_pct:.1f}%)"
    
    return True, f"Dataset valide: {df.shape[0]} lignes, {len(numeric_cols)} features numériques"

def auto_detect_target(df):
    """
    Détecte automatiquement la variable cible dans un dataset
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        str: Nom de la colonne cible ou None
    """
    # Mots-clés à rechercher
    target_keywords = ['bankrupt', 'default', 'failure', 'class', 'target', 'label', 'status']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in target_keywords):
            # Vérifier que c'est bien binaire ou catégoriel
            if df[col].nunique() <= 5:
                return col
    
    return None