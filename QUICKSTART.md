# Quickstart - Lancer le Projet

Suivez ces étapes pour configurer et lancer le projet de détection d'anomalies financières.

## 1. Prérequis

Assurez-vous d'avoir Python installé (recommandé : 3.10+). 

> [!IMPORTANT]
> Ce projet nécessite **Numpy 2.2.6+** et **Scikit-Learn 1.7.2+** pour le chargement correct des modèles pré-entraînés.

## 2. Installation des dépendances

Installez les bibliothèques nécessaires à l'aide de `pip` :

```bash
pip install -r requirements.txt
```

## 3. Lancer l'Application Streamlit

L'application Streamlit est le point d'entrée principal pour l'utilisateur final. Elle utilise les modules Python situés dans les dossiers `processing`, `analysis` et `scraping`.

Pour la lancer :
```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible par défaut à l'adresse `http://localhost:8501`.

## 4. Utiliser les Notebooks (Exploration)

Si vous souhaitez voir le détail des analyses ou expérimenter sur les données, vous pouvez lancer les notebooks dans l'ordre :

1. `01_exploration.ipynb` : Analyse exploratoire.
2. `02_cleaning.ipynb` : Nettoyage et gestion des valeurs aberrantes.
3. `03_feature_selection.ipynb` : Sélection des variables importantes.
4. `04_dimensionality_reduction.ipynb` : Analyse par composantes principales (PCA).
5. `05_modeling.ipynb` : Entraînement et comparaison des modèles.

Pour lancer l'interface Jupyter :
```bash
jupyter notebook
```

## 5. Exécuter les Scripts Python (Automatisation)

La logique des notebooks a été extraite dans des modules Python réutilisables. Vous pouvez importer ces modules dans vos propres scripts.

Par exemple, pour ré-entraîner un modèle via le script :
```python
from analysis.modeling import ModelTrainer
import pandas as pd

df = pd.read_csv('data/processed/financial_data_cleaned.csv')
trainer = ModelTrainer()
# ... suite de l'entraînement
```

## 6. Structure des Commandes

| Action | Commande |
| :--- | :--- |
| **Installer** | `pip install -r requirements.txt` |
| **Lancer le Dashboard** | `streamlit run app/streamlit_app.py` |
| **Exploration Interactive** | `jupyter notebook` |
| **Récupérer les données** | `python scraping/run_scraping.py` |

## 7. Génération de Rapports

Une fois vos prédictions effectuées dans l'application :
1. Allez dans la section **"📄 Rapport d'Expert"**.
2. Vous y trouverez un récapitulatif de vos analyses de session.
3. Cliquez sur **"📥 Télécharger le Rapport PDF"** pour obtenir un dossier complet incluant les graphiques de performance.
