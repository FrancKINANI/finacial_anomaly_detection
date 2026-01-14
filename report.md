# Rapport de Projet : Détection des Faillites d'Entreprises
## Module : Analyse de Données & Machine Learning

---

## 1. Introduction : L'Analogie du Médecin
Imaginez qu'une entreprise est comme une personne. Pour savoir si une personne est en bonne santé, un médecin prend son pouls, sa tension et fait des analyses de sang. En finance, nous faisons la même chose : nous utilisons des **ratios financiers** (le pouls de l'entreprise) pour prédire si elle risque de "tomber malade" (faire faillite).

**Objectif du projet** : Construire un "Stéthoscope Intelligent" (un modèle mathématique) capable de dire, à partir de chiffres, si une entreprise va survivre ou s'effondrer.

---

## 2. Premier Principe : La Donnée (Le Carburant)
Pour apprendre, notre intelligence artificielle a besoin d'exemples. Nous avons utilisé des données d'entreprises taïwanaises.

- **Le défi du déséquilibre** : Dans le monde réel, heureusement, peu d'entreprises font faillite. Dans nos données, nous avons 97% d'entreprises saines et seulement 3% de faillites. 
- **Le problème** : Si on donne cela tel quel à l'ordinateur, il va devenir "paresseux" et dire que tout le monde est sain pour avoir raison 97% du temps.
- **La solution (SMOTE)** : Nous avons utilisé une technique appelée **SMOTE** qui crée des "exemples synthétiques" d'entreprises en faillite pour équilibrer la balance et forcer l'ordinateur à apprendre les signes du danger.

---

## 3. Étape 1 : Le Nettoyage (La Préparation)
Avant de cuisiner, on lave les légumes. Ici, c'est le **Prétraitement**.

- **Les Outliers (Valeurs Aberrantes)** : Parfois, un chiffre est tellement énorme qu'il fausse tout (ex: une erreur de frappe). Nous utilisons la **Winsorisation** : on coupe les extrêmes pour ramener tout le monde dans une zone raisonnable.
- **La Normalisation (RobustScaler)** : Certaines données sont en millions, d'autres en pourcentages. Pour que l'ordinateur ne pense pas qu'un gros chiffre est forcément plus important, on remet tout à la même échelle (entre -1 et 1 par exemple).

---

## 4. Étape 2 : L'Analyse Exploratoire (Comprendre les Liens)
Nous cherchons quelles "maladies" mènent à la faillite.

- **Corrélation** : Si deux ratios disent exactement la même chose, nous n'avons pas besoin des deux. C'est comme mesurer la température en Celsius et en Fahrenheit : une seule suffit.
- **Visualisation** : Nous avons dessiné des graphiques pour voir où les entreprises saines et les entreprises malades se séparent.

---

## 5. Étape 3 : La Sélection (Garder l'Essentiel)
On ne peut pas regarder 100 chiffres à la fois. C'est la **Réduction de Dimension**.

- **PCA (Analyse en Composantes Principales)** : C'est comme prendre une photo 2D d'une statue en 3D. On perd un peu de détail, mais on voit l'essentiel. Cela nous aide à voir des groupes (clusters) d'entreprises.
- **Feature Selection** : Nous ne gardons que les variables qui ont un "poids" réel sur la survie.

---

## 6. Étape 4 : La Modélisation (L'Apprentissage)
Nous avons testé plusieurs "cerveaux" (algorithmes) :

1. **Régression Logistique** : Une simple ligne qui sépare les bons des mauvais. (Simple mais souvent trop rigide).
2. **Forêt Aléatoire (Random Forest)** : C'est comme demander l'avis à 100 experts (arbres de décision) et prendre la majorité. C'est très robuste.
3. **Gradient Boosting (Le Gagnant)** : Une équipe où chaque membre essaie de corriger les erreurs du précédent. C'est notre modèle le plus précis.

---

## 7. Critères de Succès (Pourquoi ça marche ?)
Nous n'utilisons pas seulement la **Précision** (Accuracy). 
- **Le Rappel (Recall)** : C'est notre priorité. Il vaut mieux se tromper en disant qu'une entreprise saine est en danger (fausse alerte), plutôt que de dire qu'une entreprise qui va mourir est en pleine forme (danger non détecté).

---

## 8. Conclusion & Déploiement
Nous avons transformé ces maths complexes en une **Application Web (Streamlit)** facile à utiliser. 
- L'utilisateur entre ses chiffres.
- Le modèle calcule le risque.
- Un rapport PDF est généré pour expliquer la décision.

**C'est la rencontre entre la finance traditionnelle et l'intelligence artificielle moderne.**
