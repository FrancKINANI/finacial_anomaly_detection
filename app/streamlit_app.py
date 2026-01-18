"""
Application Streamlit - Détection d'Anomalies Financières
Version Professionnelle Améliorée
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import requests
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
from datetime import datetime
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
import plotly.figure_factory as ff

# Ajouter la racine du projet au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scraping import run_scraping as scraper
from processing import cleaning_and_transform as preprocessor
from report import pdf_generator

# Configuration
st.set_page_config(
    page_title="Détection Anomalies Financières",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

    :root {
        --primary: #6366f1;
        --secondary: #a855f7;
        --accent: #ec4899;
    }

    .stApp {
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.1), rgba(15, 23, 42, 0.02));
    }

    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    [data-testid="stSidebarNav"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: -0.02em;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        overflow: visible;
    }
    
    [data-theme="light"] .glass-card {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        color: #1e293b;
    }
    
    [data-theme="dark"] .glass-card {
        color: #f8fafc;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(5px);
        padding: 1.5rem !important;
    }

    [data-theme="light"] .stMetric {
        background: rgba(0, 0, 0, 0.02) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
    }

    /* Customizing Streamlit Widgets */
    .stButton>button {
        border-radius: 12px;
        padding: 0.6rem 2rem;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border: none;
        color: white;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .step-badge {
        background: rgba(99, 102, 241, 0.1);
        color: #818cf8;
        padding: 0.2rem 0.8rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Chemins
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'app' else SCRIPT_DIR

@st.cache_resource
def load_model():
    """Charge le modèle ML"""
    model_path = BASE_DIR / 'models' / 'best_model.pkl'
    try:
        if not model_path.exists():
            st.warning(f"Modèle non trouvé à : {model_path}")
            return None
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_resource
def load_features():
    """Charge les noms des features"""
    features_path = BASE_DIR / 'models' / 'feature_names.pkl'
    try:
        if not features_path.exists():
            return None
        with open(features_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement des features : {e}")
        return None

@st.cache_data
def load_dataset(file_path):
    """Charge un dataset"""
    try:
        return pd.read_csv(file_path)
    except:
        return None

# Initialisation session state
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Header
st.markdown('<h1 class="main-header">📊 Détection d\'Anomalies Financières</h1>', 
            unsafe_allow_html=True)

# Récupération globale du dataset
df = st.session_state.current_dataset

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Diagnostic de version
    st.sidebar.markdown("---")
    st.sidebar.caption("🛠️ Environnement Runtime")
    st.sidebar.text(f"Numpy: {np.__version__}")
    st.sidebar.text(f"Sklearn: {sklearn.__version__}")
    st.sidebar.markdown("---")
    
    # Sélection du dataset
    st.subheader("📂 Source de Données")
    data_source = st.radio(
        "Choisir la source:",
        ["Dataset Local", "Scraper URL", "Upload CSV"],
        label_visibility="collapsed"
    )
    
    if data_source == "Dataset Local":
        datasets = {
            "Taiwanese Bankruptcy": ROOT_DIR / 'data' / 'raw' / 'data1.csv',
            "Financial Distress": ROOT_DIR / 'data' / 'raw' / 'Financial Distress.csv',
            "Cleaned Data": ROOT_DIR / 'data' / 'processed' / 'financial_data_cleaned.csv'
        }
        
        selected_dataset = st.selectbox("Dataset:", list(datasets.keys()))
        
        if st.button("📥 Charger Dataset"):
            df = load_dataset(datasets[selected_dataset])
            if df is not None:
                st.session_state.current_dataset = df
                st.success(f"✅ {selected_dataset} chargé ({df.shape[0]} lignes)")
            else:
                st.error("❌ Erreur de chargement")
    
    elif data_source == "Scraper URL":
        url = st.text_input("URL du dataset (CSV ou Page HTML):")
        is_dynamic = st.checkbox("Page Dynamique (Playwright)", help="Utilisez cette option pour les sites utilisant beaucoup de JavaScript")
        if st.button("🌐 Lancer le Scraping"):
            with st.spinner("Scraping en cours (cette étape peut prendre 10-60s)..."):
                try:
                    df = scraper.scrape_from_url(url, dynamic=is_dynamic)
                    if df is not None:
                        st.session_state.current_dataset = df
                        st.success(f"✅ Données récupérées ({df.shape[0]} lignes)")
                    else:
                        st.error("❌ Échec du scraping. Vérifiez l'URL ou tentez le mode dynamique.")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    else:  # Upload CSV
        uploaded = st.file_uploader("Choisir un fichier CSV:", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.current_dataset = df
            st.success(f"✅ Fichier chargé ({df.shape[0]} lignes)")
    
    st.markdown("---")
    
    # Arrêt si aucun dataset n'est chargé
    if df is None:
        st.info("👋 Bienvenue ! Veuillez charger un dataset dans le menu à gauche pour commencer l'analyse.")
        st.stop()

    # Définition des colonnes (Global)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Navigation Restructurée
    st.subheader("🚀 Étapes de l'Analyse")
    page = st.radio(
        "Navigation:",
        [
            "🏠 Vue d'ensemble", 
            "🔍 Exploration (EDA)", 
            "⚙️ Transformation",
            "🧹 Nettoyage des Données", 
            "🎯 Sélection & Ingénierie",
            "📉 Réduction (MCA/PCA/AFD)",
            "🤖 Évaluation Modèles",
            "🔮 Prédiction du Risque", 
            "📄 Rapport d'Expert"
        ],
        label_visibility="collapsed"
    )

# ============================================================================
# RÉCUPÉRATION DES DONNÉES & MODÈLES
# ============================================================================

model = load_model()
features = load_features()
df = st.session_state.current_dataset

if df is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 5rem;">
        <h2 style="color: #818cf8;">👋 Bienvenue dans l'Expertise Financière</h2>
        <p>Veuillez charger un dataset depuis la barre latérale pour commencer l'analyse.</p>
        <div style="font-size: 5rem; margin: 2rem 0;">📊</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# ROUTING DES PAGES
# ============================================================================

# 1. ACCUEIL / VUE D'ENSEMBLE
if page == "🏠 Vue d'ensemble":
    st.markdown('<h1 class="main-header">Vue d\'Ensemble du Projet</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Lignes", f"{len(df):,}")
    with col2:
        st.metric("Total Colonnes", f"{len(df.columns)}")
    with col3:
        st.metric("Valeurs Manquantes", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Modèle IA", "RF Active" if model else "En attente")

    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Objectif du Projet</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">
            Ce système utilise des algorithmes de Machine Learning de pointe pour prédire le risque de faillite d'une entreprise 
            sur la base de ratios financiers complexes. Notre objectif est de transformer les données brutes en insights 
            actionnables pour les analystes financiers et les gestionnaires de risques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        <span class="step-badge">Scikit-Learn</span>
        <span class="step-badge">SMOTE</span>
        <span class="step-badge">Random Forest</span>
        <span class="step-badge">PCA</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("💡 Aperçu des Données")
    st.dataframe(df.head(10), width='stretch')

# 2. EDA
elif page == "🔍 Exploration (EDA)":
    st.markdown('<h1 class="main-header">Exploration Statistique</h1>', unsafe_allow_html=True)
    
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        plot_type = st.selectbox("Type de Graphique:", 
            ["📊 Barres (Catégoriel)", "📈 Histogramme & KDE", "🎻 Violin Plot", "📦 Box Plot", "🔥 Heatmap de Corrélation", "🌌 Scatter Matrix"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if plot_type == "📊 Barres (Catégoriel)":
        options = cat_cols + (['Bankrupt?'] if 'Bankrupt?' in df.columns and 'Bankrupt?' not in cat_cols else [])
        if options:
            col_bar = st.selectbox("Sélectionner la variable:", options)
            if col_bar:
                counts = df[col_bar].value_counts().reset_index()
                counts.columns = ['Valeur', 'Nombre']
                fig = px.bar(counts, x='Valeur', y='Nombre', 
                            color='Valeur', color_discrete_sequence=px.colors.sequential.Purples)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("⚠️ Aucune variable catégorielle disponible pour ce graphique.")

    elif plot_type == "📈 Histogramme & KDE":
        col1 = st.selectbox("Variable:", numeric_cols)
        fig = px.histogram(df, x=col1, marginal="rug", color_discrete_sequence=['#6366f1'])
        st.plotly_chart(fig, width='stretch')

    elif plot_type == "🎻 Violin Plot":
        col1 = st.selectbox("Variable Numérique:", numeric_cols)
        col2 = st.selectbox("Grouper par (Optionnel):", [None] + cat_cols + ['Bankrupt?'])
        fig = px.violin(df, y=col1, x=col2, box=True, points="all", color_discrete_sequence=['#a855f7'])
        st.plotly_chart(fig, width='stretch')

    elif plot_type == "📦 Box Plot":
        col1 = st.selectbox("Variable Numérique:", numeric_cols)
        fig = px.box(df, y=col1, color_discrete_sequence=['#ec4899'])
        st.plotly_chart(fig, width='stretch')

    elif plot_type == "🔥 Heatmap de Corrélation":
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, width='stretch')

    elif plot_type == "🌌 Scatter Matrix":
        selected_cols = st.multiselect("Colonnes (max 4):", numeric_cols, default=numeric_cols[:3])
        if len(selected_cols) > 1:
            fig = px.scatter_matrix(df, dimensions=selected_cols, color='Bankrupt?' if 'Bankrupt?' in df.columns else None)
            st.plotly_chart(fig, width='stretch')

# ⚙️ TRANSFORMATION
elif page == "⚙️ Transformation":
    st.markdown('<h1 class="main-header">Préparation des Données</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧬 Encodage Catégoriel")
        encoding_type = st.radio("Méthode:", ["Aucune", "One-Hot Encoding", "Label Encoding"])
        cols_to_encode = st.multiselect("Variables à encoder:", cat_cols)
        
        if st.button("Appliquer Encodage") and cols_to_encode:
            if encoding_type == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=cols_to_encode)
                st.session_state.current_dataset = df
                st.success("One-Hot Encoding appliqué !")
                st.write("### 📸 Nouvelles Variables")
                new_cols = [c for c in df.columns if any(orig in c for orig in cols_to_encode)]
                fig = px.bar(x=new_cols[:15], y=[1]*len(new_cols[:15]), title="Aperçu des nouvelles colonnes created")
                st.plotly_chart(fig, width='stretch')
            elif encoding_type == "Label Encoding":
                le = LabelEncoder()
                for col in cols_to_encode:
                    df[col] = le.fit_transform(df[col].astype(str))
                st.session_state.current_dataset = df
                st.success("Label Encoding appliqué !")
                st.write(f"### 📊 Distribution : {cols_to_encode[0]}")
                fig = px.histogram(df, x=cols_to_encode[0], color_discrete_sequence=['#818cf8'])
                st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚖️ Mise à l'échelle (Scaling)")
        scaling_type = st.radio("Méthode:", ["Aucune", "Standardisation", "Normalisation", "Robust Scaling"])
        cols_to_scale = st.multiselect("Variables à scaler:", numeric_cols)
        
        if st.button("Appliquer Scaling") and cols_to_scale:
            # Sauvegarder l'original pour la comparaison
            original_data = df[cols_to_scale[0]].copy() if cols_to_scale else None
            
            if scaling_type == "Standardisation":
                scaler = StandardScaler()
            elif scaling_type == "Normalisation":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            st.session_state.current_dataset = df
            st.success(f"{scaling_type} appliqué !")
            
            if original_data is not None:
                st.write(f"### 📉 Comparaison : {cols_to_scale[0]}")
                comp_df = pd.DataFrame({
                    'Original': (original_data - original_data.mean()) / original_data.std() if scaling_type=="Standardisation" else original_data,
                    'Transformé': df[cols_to_scale[0]]
                })
                fig = px.box(df, y=cols_to_scale[0], title=f"Nouvelle distribution de {cols_to_scale[0]}", color_discrete_sequence=['#f472b6'])
                st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Aperçu des Données Actuelles")
    st.dataframe(df.head(10), width='stretch')
    st.write(f"Types de colonnes : {df.dtypes.value_counts().to_dict()}")
    st.markdown('</div>', unsafe_allow_html=True)

# 3. NETTOYAGE
elif page == "🧹 Nettoyage des Données":
    st.markdown('<h1 class="main-header">Nettoyage & Qualité</h1>', unsafe_allow_html=True)
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        st.write("### 🚨 Valeurs Manquantes Détectées")
        st.bar_chart(null_counts[null_counts > 0])
    else:
        st.success("✨ Aucune valeur manquante détectée ! Le dataset est propre.")
    
    st.subheader("🔍 Aperçu du Dataset Nettoyé")
    st.dataframe(df.head(15), width='stretch')
    
    st.subheader("🛠️ Techniques Appliquées")
    st.markdown("""
    - **Imputation** : Remplacement des valeurs manquantes par la médiane.
    - **Outliers** : Détection via l'Intervalle Interquartile (IQR).
    - **Normalisation** : Utilisation de RobustScaler pour gérer les valeurs extrêmes.
    """)

# 4. SÉLECTION & INGÉNIERIE
elif page == "🎯 Sélection & Ingénierie":
    st.markdown('<h1 class="main-header">Feature Engineering</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🛠️ Ingénierie", "🎯 Sélection"])
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🏗️ Création de Variables")
        new_feat_name = st.text_input("Nom de la nouvelle variable:", "feat_agg")
        feat_to_agg = st.multiselect("Variables à agréger (Moyenne):", numeric_cols)
        if st.button("Créer Variable") and feat_to_agg:
            df[new_feat_name] = df[feat_to_agg].mean(axis=1)
            st.session_state.current_dataset = df
            st.success(f"Variable '{new_feat_name}' créée !")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🔍 Importance des Variables")
        if 'Bankrupt?' in df.columns:
            X = df.select_dtypes(include=[np.number]).drop('Bankrupt?', axis=1).fillna(0)
            y = df['Bankrupt?']
            rf = RandomForestClassifier(n_estimators=50)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig = px.bar(x=importances.values, y=importances.index, orientation='h', title="Top 15 variables (RF Importance)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Cible 'Bankrupt?' absente pour la sélection supervisée.")
        st.markdown('</div>', unsafe_allow_html=True)

# 5. RÉDUCTION (MCA/PCA/AFD)
elif page == "📉 Réduction (MCA/PCA/AFD)":
    st.markdown('<h1 class="main-header">Analyse Factorielle</h1>', unsafe_allow_html=True)
    
    method = st.selectbox("Méthode de Réduction:", ["PCA (Numérique)", "LDA/AFD (Supervisé)", "MCA (Catégoriel - Approximé)"])
    
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    
    if method == "PCA (Numérique)":
        if len(numeric_df.columns) > 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(StandardScaler().fit_transform(numeric_df))
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            if 'Bankrupt?' in df.columns: pca_df['Target'] = df['Bankrupt?'].values
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target' if 'Target' in pca_df.columns else None, title="Projection PCA (2D)")
            st.plotly_chart(fig, width='stretch')
            st.subheader("📋 Composantes Principales (Aperçu)")
            st.dataframe(pca_df.head(10), width='stretch')

    elif method == "LDA/AFD (Supervisé)":
        if 'Bankrupt?' in df.columns and len(numeric_df.columns) > 2:
            X = numeric_df.drop('Bankrupt?', axis=1) if 'Bankrupt?' in numeric_df.columns else numeric_df
            y = df['Bankrupt?']
            lda = LDA(n_components=1)
            components = lda.fit_transform(X, y)
            lda_df = pd.DataFrame(data=components, columns=['LD1'])
            lda_df['Target'] = y.values
            fig = px.histogram(lda_df, x='LD1', color='Target', barmode='overlay', title="Projection LDA (Séparation des Classes)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.error("LDA nécessite une variable cible et plusieurs variables numériques.")
    
    elif method == "MCA (Catégoriel - Approximé)":
        st.info("L'Analyse des Correspondances Multiples (MCA) est ici approximée par une PCA sur des variables indicatrices (Dummy variables).")
        cat_df = df.select_dtypes(exclude=[np.number])
        if not cat_df.empty:
            dummy_df = pd.get_dummies(cat_df)
            pca = PCA(n_components=2)
            comp = pca.fit_transform(dummy_df)
            mca_df = pd.DataFrame(comp, columns=['Factor 1', 'Factor 2'])
            if 'Bankrupt?' in df.columns: mca_df['Target'] = df['Bankrupt?'].values
            fig = px.scatter(mca_df, x='Factor 1', y='Factor 2', color='Target' if 'Target' in mca_df.columns else None, title="Projection MCA (Dummy PCA)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Aucune variable catégorielle trouvée pour la MCA.")

# 6. ÉVALUATION MODÈLES
elif page == "🤖 Évaluation Modèles":
    st.markdown('<h1 class="main-header">Performance du Modèle</h1>', unsafe_allow_html=True)
    
    # Charger les métriques
    try:
        metrics_df = pd.read_csv(BASE_DIR / 'models' / 'model_metrics.csv')
        best_row = metrics_df.iloc[0]
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Precision", f"{best_row['Precision']:.2f}")
        m_col2.metric("Recall", f"{best_row['Recall']:.2f}")
        m_col3.metric("F1-Score", f"{best_row['F1-Score']:.2f}")
        m_col4.metric("Accuracy", f"{best_row['Accuracy']:.2f}")
        m_col5.metric("ROC-AUC", f"{best_row['ROC-AUC']:.2f}")
    except:
        st.warning("Métriques détaillées non disponibles.")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📈 Matrice de Confusion")
        conf_mat_path = BASE_DIR / 'data' / 'figures' / 'confusion_matrix.png'
        if conf_mat_path.exists():
            st.image(str(conf_mat_path), use_container_width=True)
        else:
            st.info("Visualisation de la matrice non disponible.")
        
    with col2:
        st.write("### 📉 Courbe ROC")
        roc_path = BASE_DIR / 'data' / 'figures' / 'roc_pr_curves.png'
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("Visualisation ROC non disponible.")

    st.markdown("---")
    st.write("### 📊 Comparaison des Modèles")
    comp_path = BASE_DIR / 'data' / 'figures' / 'models_comparison.png'
    if comp_path.exists():
        st.image(str(comp_path), use_container_width=True)

# 7. PRÉDICTION
elif page == "🔮 Prédiction du Risque":
    st.markdown('<h1 class="main-header">Moteur de Prédiction</h1>', unsafe_allow_html=True)
    
    if model is None or features is None:
        st.error("⚠️ Modèle non disponible.")
        st.stop()
        
    tab1, tab2 = st.tabs(["📝 Saisie Manuelle", "📋 Depuis le Dataset"])
    
    with tab1:
        input_data = {}
        cols = st.columns(3)
        for idx, f in enumerate(features[:9]):
            with cols[idx % 3]:
                input_data[f] = st.number_input(f.strip()[:30], value=0.0, format="%.4f")
        for f in features:
            if f not in input_data: input_data[f] = 0.0

    with tab2:
        row_idx = st.number_input("Index ligne:", 0, len(df)-1, 0)
        row = df.iloc[row_idx]
        input_data = {f: float(row[f]) if f in row.index else 0.0 for f in features}
        st.dataframe(row.to_frame().T)

    if st.button("🚀 LANCER L'ANALYSE IA", width='stretch'):
        input_df = pd.DataFrame([input_data])[features]
        prob = model.predict_proba(input_df)[0][1]
        
        # Sauvegarder dans l'historique
        st.session_state.predictions_history.append({
            'timestamp': datetime.now(),
            'prediction': 1 if prob > 0.5 else 0,
            'probability': float(prob),
            'features': input_data
        })
        
        st.markdown("---")
        colA, colB = st.columns([1, 1])
        with colA:
            if prob > 0.5:
                st.markdown(f'<div style="background: rgba(236, 72, 153, 0.2); padding: 2rem; border-radius: 20px; border: 1px solid #ec4899; text-align: center;">'
                           f'<h2 style="color: #f472b6;">RISQUE ÉLEVÉ</h2>'
                           f'<h1 style="font-size: 4rem;">{prob*100:.1f}%</h1></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background: rgba(99, 102, 241, 0.2); padding: 2rem; border-radius: 20px; border: 1px solid #6366f1; text-align: center;">'
                           f'<h2 style="color: #818cf8;">ENTREPRISE SAINE</h2>'
                           f'<h1 style="font-size: 4rem;">{(1-prob)*100:.1f}%</h1></div>', unsafe_allow_html=True)
        
        with colB:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, 
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ec4899"}}))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, width='stretch')

# 8. RAPPORT
elif page == "📄 Rapport d'Expert":
    st.markdown('<h1 class="main-header">Rapport Professionnel</h1>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.info("Aucune prédiction effectuée pour le moment. Réalisez des analyses dans la section 'Prédiction' pour générer un rapport.")
    else:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3>Générer le Dossier d'Analyse</h3>
            <p>Téléchargez un rapport complet incluant {len(st.session_state.predictions_history)} analyse(s) effectuée(s).</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_data = pdf_generator.generate_report(
            st.session_state.predictions_history,
            dataset=st.session_state.current_dataset
        )
        
        st.download_button(
            label="📥 Télécharger le Rapport PDF",
            data=pdf_data,
            file_name=f"rapport_expertise_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="download-report"
        )
        
        st.write("### 🕒 Historique Récent")
        hist_df = pd.DataFrame([
            {
                'Heure': p['timestamp'].strftime('%H:%M:%S'),
                'Résultat': "⚠️ Risque" if p['prediction'] == 1 else "✅ Sain",
                'Confiance': f"{p['probability']*100:.1f}%" if p['prediction'] == 1 else f"{(1-p['probability'])*100:.1f}%"
            } for p in st.session_state.predictions_history[::-1][:5]
        ])
        st.table(hist_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; padding: 2rem; font-family: 'Outfit', sans-serif;">
    <p>© 2026 AI Financial Intelligence - Système de Détection d'Anomalies Alpha</p>
</div>
""", unsafe_allow_html=True)