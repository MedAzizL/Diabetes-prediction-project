import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Diabète",
    page_icon="🏥",
    layout="wide"
)

# Style CSS
st.markdown("""
<style>
* {
    font-family: 'Segoe UI', Arial, sans-serif;
}

.header {
    background: linear-gradient(to right, #6A36FC, #8B5CF6);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.card {
    background-color: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
}

.high-risk {
    background-color: rgba(239, 68, 68, 0.1);
    border: 2px solid #EF4444;
}

.high-risk h2 {
    color: #EF4444;
}

.low-risk {
    background-color: rgba(16, 185, 129, 0.1);
    border: 2px solid #10B981;
}

.low-risk h2 {
    color: #10B981;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1rem 0;
    padding-left: 10px;
    border-left: 4px solid #6A36FC;
}

/* Styliser les métriques */
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #6A36FC !important;
}

/* Styliser les tabs */
button[data-baseweb="tab"] {
    background-color: #f3f4f6;
    border-radius: 8px;
    margin-right: 8px;
    padding: 8px 16px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #6A36FC;
    color: white;
}

/* Styliser les sliders */
.stSlider label {
    font-weight: 500 !important;
    color: #374151 !important;
}

.stSlider [data-baseweb="thumb"] {
    background-color: #6A36FC;
}

/* Masquer les messages d'avertissement de label vide */
[data-testid="stStatusWidget"] {
    display: none !important;
}

.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 2rem;
    background: rgba(0,0,0,0.05);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="header">
    <h1>🏥 Prédiction du Diabète</h1>
    <p>Analysez vos facteurs de risque et obtenez une évaluation personnalisée</p>
</div>
""", unsafe_allow_html=True)

# Création des onglets
tabs = st.tabs(["Prédiction", "Caractéristiques", "À propos"])

# Fonction pour charger ou créer le modèle
@st.cache_resource
def load_model(model_type="RandomForest"):
    model_path = f'model_{model_type.lower()}.pkl'
    scaler_path = 'scaler.pkl'
    
    # Si les modèles n'existent pas, on les entraîne
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        with st.spinner(f"Entraînement du modèle {model_type} en cours..."):
            # Charger les données
            df = pd.read_csv('../data/diabetes.csv')
            
            # Prétraitement
            df_clean = df.copy()
            for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
                df_clean[column] = df_clean[column].replace(0, np.nan)
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            
            # Diviser en X et y
            X = df_clean.drop('Outcome', axis=1)
            y = df_clean['Outcome']
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Sélection du modèle
            if model_type == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "LogisticRegression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "SVM":
                model = SVC(probability=True, random_state=42)
            else:
                model = RandomForestClassifier(random_state=42)
                
            # Entraînement du modèle
            model.fit(X_scaled, y)
            
            # Sauvegarde du modèle et du scaler
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            if not os.path.exists(scaler_path):
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
    
    # Charger les modèles sauvegardés
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Configuration de la sidebar
with st.sidebar:
    st.markdown('<div class="section-title">Configuration du modèle</div>', unsafe_allow_html=True)
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Algorithme de prédiction",
        ["RandomForest", "LogisticRegression", "SVM"]
    )
    
    # Charger le modèle
    model, scaler = load_model(model_type)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Données du patient</div>', unsafe_allow_html=True)
    
    # Paramètres par groupe
    with st.expander("Informations démographiques", expanded=True):
        pregnancies = st.slider(
            "Nombre de grossesses",
            min_value=0, max_value=15, value=3, step=1
        )
        
        age = st.slider(
            "Âge", 
            min_value=21, max_value=81, value=33, step=1
        )
    
    with st.expander("Mesures glycémiques", expanded=True):
        glucose = st.slider(
            "Concentration en glucose (mg/dL)",
            min_value=70, max_value=200, value=120, step=1
        )
        
        insulin = st.slider(
            "Insuline (µU/ml)",
            min_value=10, max_value=200, value=79, step=1
        )
    
    with st.expander("Mesures physiques", expanded=True):
        blood_pressure = st.slider(
            "Pression artérielle (mm Hg)",
            min_value=60, max_value=120, value=70, step=1
        )
        
        skin_thickness = st.slider(
            "Épaisseur de la peau (mm)",
            min_value=10, max_value=50, value=20, step=1
        )
        
        bmi = st.slider(
            "Indice de masse corporelle (kg/m²)",
            min_value=18.5, max_value=40.0, value=25.0, step=0.1,
            format="%.1f"
        )
    
    with st.expander("Facteurs héréditaires", expanded=True):
        dpf = st.slider(
            "Fonction de pedigree du diabète",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            format="%.1f"
        )

# Préparer les données pour la prédiction
user_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}

# Convertir en DataFrame et faire la prédiction
user_input = pd.DataFrame(user_data, index=[0])
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Onglet Prédiction
with tabs[0]:
    # Disposition en deux colonnes
    col1, col2 = st.columns([1, 1])
    
    # Colonne 1: Résultat de la prédiction
    with col1:
        st.markdown('<div class="section-title">Résultat de l\'analyse</div>', unsafe_allow_html=True)
        
        # Affichage du résultat
        risk_class = "high-risk" if prediction[0] == 1 else "low-risk"
        risk_icon = "⚠️" if prediction[0] == 1 else "✅"
        risk_text = "Risque élevé de diabète" if prediction[0] == 1 else "Risque faible de diabète"
        
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <h2>{risk_icon} {risk_text}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage des probabilités
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric(
                label="Probabilité de diabète",
                value=f"{prediction_proba[0][1]:.1%}"
            )
        with col1b:
            st.metric(
                label="Probabilité d'absence",
                value=f"{prediction_proba[0][0]:.1%}"
            )
        
        # Interprétation des résultats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Interprétation</div>', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown("""
            - **Risque élevé**: Le modèle a détecté des facteurs de risque significatifs
            - Cette prédiction est basée principalement sur les valeurs de glucose, IMC et âge
            - Un suivi médical est recommandé pour une évaluation complète
            """)
        else:
            st.markdown("""
            - **Risque faible**: Le modèle n'a pas détecté de facteurs de risque significatifs
            - Maintenir un mode de vie sain et des contrôles médicaux réguliers est recommandé
            - Cette prédiction est indicative et non un diagnostic médical
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonne 2: Visualisation du niveau de risque
    with col2:
        st.markdown('<div class="section-title">Niveau de risque</div>', unsafe_allow_html=True)
        
        # Graphique à barres horizontales
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=['Non-diabète', 'Diabète'],
            x=[prediction_proba[0][0], prediction_proba[0][1]],
            orientation='h',
            marker_color=['#10B981', '#EF4444'],
            text=[f"{prediction_proba[0][0]:.1%}", f"{prediction_proba[0][1]:.1%}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis=dict(
                title='Probabilité',
                tickformat='.0%',
                range=[0, 1.1]
            ),
            margin=dict(l=20, r=40, t=20, b=20),
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Jauge de risque
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0][1]*100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6A36FC"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [30, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        
        fig.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Légende pour la jauge
        st.markdown("""
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #10B981; font-weight: 600;">Faible (0-30%)</span>
            <span style="color: #F59E0B; font-weight: 600;">Modéré (30-70%)</span>
            <span style="color: #EF4444; font-weight: 600;">Élevé (70-100%)</span>
        </div>
        """, unsafe_allow_html=True)

# Onglet Caractéristiques
with tabs[1]:
    st.markdown('<div class="section-title">Caractéristiques du patient</div>', unsafe_allow_html=True)
    
    # Tableau de données
    st.dataframe(
        user_input,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualisation radar
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Normaliser les données pour le radar chart
    features_norm = user_input.copy()
    ranges = {
        'Pregnancies': (0, 15),
        'Glucose': (70, 200),
        'BloodPressure': (60, 120),
        'SkinThickness': (10, 50),
        'Insulin': (10, 200),
        'BMI': (18.5, 40.0),
        'DiabetesPedigreeFunction': (0.1, 2.0),
        'Age': (21, 81)
    }
    
    for col in features_norm.columns:
        min_val, max_val = ranges[col]
        features_norm[col] = (features_norm[col] - min_val) / (max_val - min_val)
    
    # Créer le radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=features_norm.iloc[0].values,
        theta=features_norm.columns,
        fill='toself',
        fillcolor='rgba(106, 54, 252, 0.2)',
        line=dict(color='#6A36FC', width=2),
        name='Patient'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Importance des caractéristiques (si disponible)
    if hasattr(model, 'feature_importances_'):
        st.markdown('<div class="section-title">Importance des caractéristiques</div>', unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': user_input.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=350,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Onglet À propos
with tabs[2]:
    st.markdown('<div class="section-title">À propos du projet</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Prédiction du diabète
    
    Cette application utilise des algorithmes d'apprentissage automatique pour prédire le risque de diabète en se basant sur des caractéristiques médicales.
    
    **Objectifs du projet**:
    - Développer un modèle prédictif pour identifier les personnes à risque de diabète
    - Comparer différentes approches d'apprentissage automatique
    - Offrir un outil d'aide à la décision pour les professionnels de la santé
    
    **Source des données**: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
    
    > **Note importante**: Cette application est conçue à des fins éducatives uniquement et ne remplace pas un avis médical professionnel.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Caractéristiques du modèle sélectionné
    st.markdown('<div class="section-title">Caractéristiques du modèle</div>', unsafe_allow_html=True)
    
    # Informations sur le modèle sélectionné
    model_info = {
        'RandomForest': {
            'description': "Un ensemble d'arbres de décision qui vote collectivement pour faire une prédiction.",
            'avantages': ["Robuste aux valeurs aberrantes", "Bonne performance sur divers types de données", "Capture les relations non linéaires"],
            'inconvenients': ["Moins interprétable que les modèles linéaires", "Peut souffrir de surapprentissage"]
        },
        'LogisticRegression': {
            'description': "Un modèle linéaire classique qui estime la probabilité d'appartenance à une classe.",
            'avantages': ["Simple et interprétable", "Efficace pour les relations linéaires", "Moins sujet au surapprentissage"],
            'inconvenients': ["Ne capture pas les relations complexes", "Moins performant sur certaines données"]
        },
        'SVM': {
            'description': "Un modèle qui trouve un hyperplan optimal pour séparer les classes dans un espace de haute dimension.",
            'avantages': ["Efficace dans les espaces de haute dimension", "Versatile grâce aux noyaux", "Robuste"],
            'inconvenients': ["Difficile à interpréter", "Sensible au choix des hyperparamètres"]
        }
    }
    
    current_model = model_info[model_type]
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"Modèle: {model_type}")
    st.markdown(f"**Description**: {current_model['description']}")
    
    st.markdown("**Avantages**:")
    for adv in current_model['avantages']:
        st.markdown(f"- {adv}")
    
    st.markdown("**Limitations**:")
    for inc in current_model['inconvenients']:
        st.markdown(f"- {inc}")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Développé dans le cadre d'un mini-projet d'analyse et fouille de données | 2025</p>
    <p><small>Cette application est conçue à des fins éducatives uniquement et ne constitue pas un avis médical.</small></p>
</div>
""", unsafe_allow_html=True) 