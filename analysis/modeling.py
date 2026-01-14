"""
Module de modélisation pour la détection d'anomalies financières.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles"""
    
    def __init__(self, target_col='Bankrupt?'):
        self.target_col = target_col
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""
        self.results = {}

    def prepare_data(self, df, test_size=0.2, use_smote=True):
        """Prépare les données pour l'entraînement"""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Entraîne plusieurs modèles et compare leurs performances"""
        best_auc = 0
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_prob)
            self.results[name] = {
                'auc': auc,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if auc > best_auc:
                best_auc = auc
                self.best_model = model
                self.best_model_name = name
                
        return self.results

    def save_model(self, model_path, feature_names_path):
        """Sauvegarde le meilleur modèle et les noms des features"""
        if self.best_model:
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            features = list(self.best_model.feature_names_in_)
            with open(feature_names_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Modèle '{self.best_model_name}' sauvegardé avec succès.")

    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Affiche la matrice de confusion du meilleur modèle"""
        if self.best_model is None:
            return
            
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.title(f'Matrice de Confusion - {self.best_model_name}')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
