"""
Playwright Scraper - Gestion des pages dynamiques
"""

from playwright.sync_api import sync_playwright
import pandas as pd
from io import StringIO
import time

def scrape_dynamic_csv(url, wait_time=5):
    """
    Scrape un CSV depuis une page dynamique utilisant Playwright
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Attendre un peu plus si nécessaire pour le rendu JS
            time.sleep(wait_time)
            
            # Tenter de trouver du contenu CSV dans le texte de la page
            content = page.content()
            
            # Si l'URL finit par .csv, Playwright pourrait l'afficher directement
            # ou on peut essayer d'extraire le texte brut
            text_content = page.evaluate("() => document.body.innerText")
            
            # Nettoyage basique
            if "," in text_content and "\n" in text_content:
                df = pd.read_csv(StringIO(text_content))
                browser.close()
                return df
            
            browser.close()
            return None
            
        except Exception as e:
            print(f"Erreur Playwright: {e}")
            browser.close()
            return None

def scrape_table_to_df(url, table_selector="table"):
    """
    Scrape un tableau HTML depuis une page dynamique et le convertit en DataFrame
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle")
            
            # Extraire le HTML de la table
            table_html = page.inner_html(table_selector)
            df_list = pd.read_html(StringIO(f"<table>{table_html}</table>"))
            
            browser.close()
            if df_list:
                return df_list[0]
            return None
            
        except Exception as e:
            print(f"Erreur extraction table: {e}")
            browser.close()
            return None
