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
    -   `streamlit_app.py` : Interface utilisateur complète (EDA, Transformation, Prédiction, Rapport).
-   **`report/`** : Génération de rapports.
    -   `pdf_generator.py` : Export automatisé des analyses en format PDF professionnel.
-   **`notebooks/`** : Carnets Jupyter pour l'expérimentation pas à pas.
-   **`models/`** : Stockage du meilleur modèle entraîné (`best_model.pkl`) et des métriques.

## Fonctionnalités Clés

L'application Streamlit propose une interface complète structurée en 8 étapes :

1.  **🔍 Exploration (EDA)** : Visualisations avancées (Violin plots, KDE, matrices de corrélation, Box plots).
2.  **⚙️ Transformation** : Encodage (One-Hot, Label) et Mise à l'échelle (Standard, MinMax, Robust) avec prévisualisation interactive.
3.  **🧹 Nettoyage** : Gestion des doublons et des valeurs manquantes.
4.  **🎯 Sélection & Ingénierie** : Analyse de l'importance des variables par Random Forest et agrégations.
5.  **📉 Réduction (MCA/PCA/AFD)** : Visualisation haute dimensionnelle via PCA, LDA et MCA.
6.  **🤖 Évaluation Modèles** : Tableaux de bord de performance (Matrice de Confusion, Courbes ROC) basés sur les données réelles du modèle Gradient Boosting.
7.  **🔮 Prédiction du Risque** : Moteur de prédiction en temps réel avec saisie manuelle ou sélection de dataset, et gestion d'un historique.
8.  **📄 Rapport d'Expert** : Génération instantanée d'un dossier d'expertise financier au format PDF.

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
