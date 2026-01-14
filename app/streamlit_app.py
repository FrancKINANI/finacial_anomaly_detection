"""
Application Streamlit - Détection d'Anomalies Financières
Version Professionnelle Améliorée
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# CSS personnalisé moderne
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .danger-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-card {
        background: rgba(31, 119, 180, 0.1);
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    .info-card h3 {
        color: #1f77b4;
        margin-top: 0;
    }
    .info-card ul {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Chemins
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'app' else SCRIPT_DIR

@st.cache_resource
def load_model():
    """Charge le modèle ML"""
    try:
        with open(BASE_DIR / 'models' / 'best_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_features():
    """Charge les noms des features"""
    try:
        with open(BASE_DIR / 'models' / 'feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    except:
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

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
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
        url = st.text_input("URL du dataset (CSV):")
        if st.button("🌐 Scraper"):
            with st.spinner("Scraping en cours..."):
                try:
                    df = scraper.scrape_from_url(url)
                    if df is not None:
                        st.session_state.current_dataset = df
                        st.success(f"✅ Données scrapées ({df.shape[0]} lignes)")
                    else:
                        st.error("❌ Échec du scraping")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    else:  # Upload CSV
        uploaded = st.file_uploader("Choisir un fichier CSV:", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.current_dataset = df
            st.success(f"✅ Fichier chargé ({df.shape[0]} lignes)")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("📑 Navigation")
    page = st.radio(
        "Page:",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Dashboard", "📥 Export PDF"],
        label_visibility="collapsed"
    )

# Charger modèle
model = load_model()
features = load_features()

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Dataset", 
                 f"{st.session_state.current_dataset.shape[0]:,}" if st.session_state.current_dataset is not None else "Aucun",
                 "lignes chargées")
    
    with col2:
        st.metric("🤖 Modèle", 
                 "Actif" if model else "Inactif",
                 "Random Forest")
    
    with col3:
        st.metric("📈 Précision", 
                 "~95%" if model else "N/A",
                 "ROC-AUC")
    
    st.markdown("---")
    
    st.subheader("🎯 Fonctionnalités")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔮 Prédiction Intelligente</h3>
            <ul>
                <li>Analyse en temps réel</li>
                <li>3 modes de saisie</li>
                <li>Probabilités détaillées</li>
                <li>Recommandations personnalisées</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🌐 Web Scraping</h3>
            <ul>
                <li>Import depuis URL</li>
                <li>Analyse automatique</li>
                <li>Nettoyage intelligent</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Dashboard Analytics</h3>
            <ul>
                <li>Visualisations interactives</li>
                <li>KPIs en temps réel</li>
                <li>Analyse comparative</li>
                <li>Tendances et patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📥 Export PDF</h3>
            <ul>
                <li>Rapports professionnels</li>
                <li>Graphiques inclus</li>
                <li>Historique des prédictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: PRÉDICTION
# ============================================================================

elif page == "🔮 Prédiction":
    st.header("🔮 Analyse du Risque de Faillite")
    
    if model is None or features is None:
        st.error("⚠️ Modèle non disponible. Exécutez le notebook 05_modeling.ipynb")
        st.stop()
    
    # Mode de saisie
    tab1, tab2, tab3 = st.tabs(["📝 Saisie Manuelle", "🎲 Exemple", "📋 Dataset"])
    
    with tab1:
        st.subheader("Entrez les ratios financiers")
        
        input_data = {}
        cols = st.columns(3)
        
        key_ratios = features[:9] if len(features) > 9 else features
        
        for idx, feature in enumerate(key_ratios):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    feature.strip()[:30],
                    value=0.0,
                    format="%.4f"
                )
        
        for feature in features:
            if feature not in input_data:
                input_data[feature] = 0.0
    
    with tab2:
        if st.session_state.current_dataset is not None:
            sample = st.session_state.current_dataset.sample(1).iloc[0]
            
            input_data = {}
            for feature in features:
                if feature in sample.index:
                    input_data[feature] = float(sample[feature])
                else:
                    input_data[feature] = 0.0
            
            col1, col2, col3 = st.columns(3)
            for idx, feat in enumerate(list(input_data.keys())[:9]):
                with [col1, col2, col3][idx % 3]:
                    st.metric(feat.strip()[:25], f"{input_data[feat]:.4f}")
        else:
            st.warning("📂 Aucun dataset chargé")
    
    with tab3:
        if st.session_state.current_dataset is not None:
            row_idx = st.number_input("Ligne à analyser:", 
                                     min_value=0, 
                                     max_value=len(st.session_state.current_dataset)-1,
                                     value=0)
            
            row = st.session_state.current_dataset.iloc[row_idx]
            input_data = {f: float(row[f]) if f in row.index else 0.0 for f in features}
            
            st.dataframe(row.to_frame().T)
        else:
            st.warning("📂 Aucun dataset chargé")
    
    st.markdown("---")
    
    # Prédiction
    if st.button("🚀 ANALYSER", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])[features]
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        proba_healthy = probability[0]
        proba_bankrupt = probability[1]
        
        # Sauvegarder dans l'historique
        st.session_state.predictions_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'probability': proba_bankrupt,
            'input_data': input_data
        })
        
        st.markdown("---")
        
        # Résultat
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="danger-box">
                    <h2>⚠️ RISQUE ÉLEVÉ</h2>
                    <h1>{proba_bankrupt*100:.1f}%</h1>
                    <p>Probabilité de faillite</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ ENTREPRISE SAINE</h2>
                    <h1>{proba_healthy*100:.1f}%</h1>
                    <p>Santé financière</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_bankrupt * 100,
            title={'text': "Risque de Faillite"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: DASHBOARD
# ============================================================================

elif page == "📊 Dashboard":
    st.header("📊 Dashboard Analytique")
    
    if st.session_state.current_dataset is None:
        st.warning("📂 Chargez un dataset pour afficher le dashboard")
        st.stop()
    
    df = st.session_state.current_dataset
    
    # KPIs
    st.subheader("📈 Indicateurs Clés")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
    
    with col2:
        if 'Bankrupt?' in df.columns:
            st.metric("Entreprises Saines", f"{(df['Bankrupt?']==0).sum():,}")
    
    with col3:
        if 'Bankrupt?' in df.columns:
            st.metric("Faillites", f"{(df['Bankrupt?']==1).sum():,}")
    
    with col4:
        st.metric("Features", f"{len(df.columns)}")
    
    st.markdown("---")
    
    # Visualisations
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Corrélations", "📈 Tendances"])
    
    with tab1:
        if 'Bankrupt?' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=df['Bankrupt?'].value_counts().values,
                            names=['Sain', 'Faillite'],
                            title='Distribution des Classes',
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(x=['Sain', 'Faillite'],
                            y=df['Bankrupt?'].value_counts().values,
                            title='Nombre par Classe',
                            color=df['Bankrupt?'].value_counts().values,
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect("Sélectionner features:", 
                                          numeric_cols[:10],
                                          default=numeric_cols[:5])
            
            if len(selected_cols) > 1:
                corr = df[selected_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                               title="Matrice de Corrélation",
                               color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            feature = st.selectbox("Feature à analyser:", numeric_cols)
            
            fig = px.histogram(df, x=feature, 
                              color='Bankrupt?' if 'Bankrupt?' in df.columns else None,
                              title=f"Distribution de {feature}",
                              marginal="box")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: EXPORT PDF
# ============================================================================

elif page == "📥 Export PDF":
    st.header("📥 Export des Résultats en PDF")
    
    if not st.session_state.predictions_history:
        st.warning("⚠️ Aucune prédiction à exporter. Effectuez d'abord des prédictions.")
        st.stop()
    
    st.subheader(f"📊 Historique: {len(st.session_state.predictions_history)} prédiction(s)")
    
    # Afficher l'historique
    for idx, pred in enumerate(st.session_state.predictions_history[-5:], 1):
        with st.expander(f"Prédiction #{idx} - {pred['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Résultat", 
                         "Faillite ⚠️" if pred['prediction'] == 1 else "Saine ✅")
            with col2:
                st.metric("Probabilité", f"{pred['probability']*100:.1f}%")
    
    st.markdown("---")
    
    # Options d'export
    st.subheader("⚙️ Options d'Export")
    
    include_graphs = st.checkbox("Inclure les graphiques", value=True)
    include_details = st.checkbox("Inclure les détails techniques", value=True)
    
    if st.button("📄 Générer le PDF", type="primary"):
        with st.spinner("Génération du PDF..."):
            try:
                pdf_bytes = pdf_generator.generate_report(
                    st.session_state.predictions_history,
                    st.session_state.current_dataset,
                    include_graphs=include_graphs,
                    include_details=include_details
                )
                
                st.success("✅ PDF généré avec succès!")
                
                st.download_button(
                    label="📥 Télécharger le PDF",
                    data=pdf_bytes,
                    file_name=f"rapport_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"❌ Erreur: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Détection d'Anomalies Financières - Version Professionnelle</p>
    <p>Développé avec Streamlit | Python | Machine Learning</p>
</div>
""", unsafe_allow_html=True)