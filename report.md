# Rapport de Projet : Détection des Faillites d'Entreprises
## Module : Analyse de Données & Machine Learning

---

### **Sommaire**
1. **Introduction** : Pourquoi prédire la faillite ?
2. **Premiers principes** de l’analyse de données financières
3. **Étape 1 – Analyse Exploratoire des Données (EDA)**
4. **Étape 2 – Nettoyage et Transformation des données**
5. **Étape 3 – Sélection de variables et Réduction de dimension**
6. **Étape 4 – Modélisation et Gestion du déséquilibre**
7. **Évaluation métier** : Rappel, Précision et Coût de l’erreur
8. **Déploiement** : L'Application Streamlit
9. **Conclusion**

---

### **1. Introduction : L'Analogie du Médecin**
Imaginez qu'une entreprise est comme une personne. Pour savoir si une personne est en bonne santé, un médecin prend son pouls, sa tension et fait des analyses de sang. En finance, nous faisons la même chose : nous utilisons des **ratios financiers** (le pouls de l'entreprise) pour prédire si elle risque de "tomber malade" (faire faillite).

**Objectif du projet** : Construire un "Stéthoscope Intelligent" capable de transformer des ratios financiers complexes en une alerte précoce. Nous ne cherchons pas à remplacer l’expert financier, mais à lui offrir un outil qui amplifie sa perception des signaux faibles.

---

### **2. Premiers principes de l’analyse financière**
Avant tout modèle, il faut comprendre la nature des données. Les données financières ne sont pas des nombres aléatoires ; elles obéissent à des lois comptables.
*   **Principe de Causalité** : Un ratio de liquidité faible n'est pas juste un "petit chiffre", c'est une incapacité physique de l'entreprise à payer ses dettes.
*   **Principe de Comparabilité** : On ne compare pas des millions de dollars avec des pourcentages sans les ramener à une échelle commune.

---

### **3. Étape 1 – Analyse Exploratoire (EDA) : Écouter les données**
Le point de départ est l'humilité : ne rien supposer.
*   **Dimensions** : Le dataset contient **6 819 entreprises** avec **96 variables** (ratios).
*   **Le Défi du Déséquilibre** : Seuls **3,23 %** des cas sont des faillites (**220 entreprises**). C'est le coeur du problème : l'ordinateur risque de devenir "paresseux" et de prédire que tout le monde est sain pour avoir raison 97 % du temps.
*   **Corrélations** : Nous avons identifié que le ratio **"Net Income to Total Assets"** (le bénéfice net par rapport aux actifs) est l'un des signaux les plus corrélés à la survie de l'entreprise.

---

### **4. Étape 2 – Nettoyage : Préparer le carburant**
Avant de cuisiner, on lave les ingrédients.
*   **Outliers (Valeurs Aberrantes)** : En finance, un chiffre extrême n'est pas forcément une erreur, mais il peut fausser les calculs. Nous avons appliqué la **Winsorisation (1er et 99e centile)** pour "raboter" les extrêmes sans perdre l'information.
*   **Normalisation (RobustScaler)** : Comme la moyenne est trop sensible aux valeurs extrêmes, nous avons utilisé une méthode basée sur la **médiane** pour mettre toutes les variables à la même échelle.

---

### **5. Étape 3 – Sélection et Réduction : Garder l’essentiel**
Avoir 95 variables crée du "bruit".
*   **Sélection (Feature Selection)** : En croisant 5 méthodes (ANOVA, Mutual Info, Random Forest Importance...), nous avons réduit le dataset de **95 à 17 variables calculées**. C'est le "Top 20" des indicateurs de faillite.
*   **Réduction (PCA)** : L'Analyse en Composantes Principales nous a permis de voir que l'information essentielle pouvait être compressée. Cela réduit les risques de surapprentissage (overfitting).

---

### **6. Étape 4 – Modélisation : L'Apprentissage**
Nous avons testé 7 types de "cerveaux" mathématiques (algorithmes).
*   **Le Sauvetage par SMOTE** : Pour corriger le déséquilibre, nous avons utilisé la technique **SMOTE** qui crée des exemples synthétiques de faillites. Cela force l'ordinateur à apprendre les signes avant-coureurs.
*   **Comparaison des Modèles** :
    *   **Logistic Regression** : Trop simple (ROC-AUC bas : 0.29).
    *   **Random Forest** : Très bon (ROC-AUC : 0.92).
    *   **Gradient Boosting (Le Gagnant)** : Le plus performant avec un **ROC-AUC de 0,93**.

---

### **7. Évaluation métier : Le Coût de l'Erreur**
En science des données, la "précision globale" (Accuracy) est trompeuse.
*   **Le Rappel (Recall)** : Notre modèle atteint **75 %**. Cela signifie qu'il détecte 3 faillites sur 4.
*   **Justification** : Pour une banque, le coût d'un **Faux Négatif** (manquer une faillite) est énorme (perte totale du prêt), alors que le coût d'un **Faux Positif** (faire une faussee alerte) est faible (simple vérification manuelle). On privilégie donc la sensibilité.

---

### **8. Déploiement : Streamlit & IA Responsable**
Nous avons transformé ce pipeline complexe en une **Application Web**.
*   **Scraping** : On peut charger des données en direct depuis une URL.
*   **Interprétabilité** : L'app montre non seulement le score de risque, mais aussi les variables qui ont influencé la décision.
*   **Rapport PDF** : Un clic suffit pour générer un rapport professionnel synthétisant l'analyse.

---

### **9. Conclusion**
Ce projet démontre que la détection de faillite n’est pas qu’une affaire de chiffres, mais une question de **méthodologie**. En respectant les "premiers principes" — de la préparation minutieuse à la gestion rigoureuse du déséquilibre — nous avons créé un outil qui ne remplace pas l'expert, mais lui donne une vision augmentée des risques de demain.
