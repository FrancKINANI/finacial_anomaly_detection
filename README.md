# Détection d'Anomalies Financières (Faillite)

Ce projet vise à prédire la faillite d'entreprises en utilisant des techniques d'apprentissage automatique (Machine Learning) sur des données financières. Il comprend une chaîne complète allant du scraping de données au déploiement d'une application interactive avec Streamlit.

## Structure du Projet

Le projet est organisé selon une structure modulaire pour faciliter la maintenance et l'évolution :

-   **`data/`** : Archivage des données.
    -   `raw/` : Données brutes non modifiées.
    -   `processed/` : Données nettoyées et transformées.
    -   `figures/` : Visualisations générées durant l'analyse.
-   **`scraping/`** : Scripts pour la récupération automatique de données.
    -   `run_scraping.py` : Script principal pour lancer la collecte.
-   **`processing/`** : Nettoyage et transformation des données.
    -   `cleaning_and_transform.py` : Prétraitement (Imputation, Outliers, Scaler).
    -   `merge_data.py` : Fusion de différents datasets.
-   **`analysis/`** : Analyse exploratoire et modélisation.
    -   `exploratory_visualization.py` : Fonctions d'EDA.
    -   `feature_selection.py` : Algorithmes de sélection de variables.
    -   `dimensionality_reduction.py` : PCA et autres techniques.
    -   `modeling.py` : Entraînement et évaluation des modèles.
-   **`app/`** : Application Streamlit.
    -   `streamlit_app.py` : Interface utilisateur pour tester le modèle.
-   **`report/`** : Génération de rapports.
    -   `pdf_generator.py` : Export des résultats en PDF.
-   **`notebooks/`** : Carnets Jupyter pour l'expérimentation pas à pas.
-   **`models/`** : Stockage du meilleur modèle entraîné (`.pkl`).

## Technologies Utilisées

-   **Python 3.10+**
-   **Pandas / Numpy 2.2.6+** : Manipulation de données (compatible avec les nouveaux BitGenerators).
-   **Scikit-Learn 1.7.2+** : Machine Learning.
-   **Imbalanced-learn (SMOTE)** : Gestion du déséquilibre des classes.
-   **Matplotlib / Seaborn** : Visualisation.
-   **Streamlit 1.53.0+** : Interface Web.
-   **ReportLab** : Génération de rapports PDF.

## Installation et Utilisation

Consultez le fichier [QUICKSTART.md](QUICKSTART.md) pour les instructions détaillées.
