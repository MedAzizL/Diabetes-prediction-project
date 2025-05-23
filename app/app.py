import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
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
body, .stApp, * { font-family: 'Inter', 'Segoe UI', 'Roboto', Arial, sans-serif !important; }

.section-title { font-size: 1.3rem; font-weight: 700; color: #6A36FC; margin: 1.2rem 0 0.7rem 0; letter-spacing: 0.5px; }

.card { background-color: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; }

.result-box { padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0; box-shadow: 0 2px 8px rgba(106,54,252,0.07); background: #f8f8ff; border: 1.5px solid #ececff; font-size: 1.2rem; }

.high-risk { background-color: rgba(239, 68, 68, 0.1); border: 2px solid #EF4444; }

.high-risk h2 { color: #EF4444; }

.low-risk { background-color: rgba(16, 185, 129, 0.1); border: 2px solid #10B981; }

.low-risk h2 { color: #10B981; }

.stInfo { background: #f3f4f6; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0; color: #374151; font-size: 1.05rem; }

.metric-label { font-weight: 600; color: #6A36FC; font-size: 1.1rem; }

.metric-value { font-size: 1.5rem; font-weight: 700; color: #374151; }

.header { background: linear-gradient(to right, #6A36FC, #8B5CF6); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }

[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 600 !important; color: #6A36FC !important; }

button[data-baseweb="tab"] { background-color: #f3f4f6; border-radius: 8px; margin-right: 8px; padding: 8px 16px; }

button[data-baseweb="tab"][aria-selected="true"] { background-color: #6A36FC; color: white; }

.stSlider label { font-weight: 500 !important; color: #374151 !important; }

.stSlider [data-baseweb="thumb"] { background-color: #6A36FC; }

[data-testid="stStatusWidget"] { display: none !important; }

.footer { text-align: center; padding: 1.5rem; margin-top: 2rem; background: rgba(0,0,0,0.05); border-radius: 10px; }
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
tabs = st.tabs(["Prédiction", "Caractéristiques"])

# Fonction pour charger ou créer le modèle
@st.cache_resource
def load_model(model_type="RandomForest"):
    model_path = f'model_{model_type.lower()}.pkl'
    scaler_path = 'scaler.pkl'
    
    # Force retraining for DBSCAN model
    if model_type == "DBSCAN" and os.path.exists(model_path):
        os.remove(model_path)
    
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
                model.fit(X_scaled, y)
            elif model_type == "LogisticRegression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_scaled, y)
            elif model_type == "SVM":
                model = SVC(probability=True, random_state=42)
                model.fit(X_scaled, y)
            elif model_type == "KMeans":
                model = KMeans(n_clusters=2, random_state=42)
                model.fit(X_scaled)
            elif model_type == "DBSCAN":
                # Paramètres pour des clusters plus larges
                model = DBSCAN(eps=2.0, min_samples=5)
                model.fit(X_scaled)
            else:
                model = RandomForestClassifier(random_state=42)
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
        ["RandomForest", "LogisticRegression", "SVM", "KMeans", "DBSCAN"]
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

# Prédiction différente selon le type de modèle
if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
    # Modèles supervisés
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
else:
    if model_type == "KMeans":
        cluster = model.predict(user_input_scaled)[0]
        df = pd.read_csv('../data/diabetes.csv')
        df_clean = df.copy()
        for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df_clean[column] = df_clean[column].replace(0, np.nan)
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        X = df_clean.drop('Outcome', axis=1)
        X_scaled = scaler.transform(X)
        clusters = model.predict(X_scaled)
        df_with_clusters = pd.DataFrame({'cluster': clusters, 'outcome': df_clean['Outcome']})
        diabetic_ratio = df_with_clusters.groupby('cluster')['outcome'].mean()
        cluster_size = df_with_clusters['cluster'].value_counts()[cluster]
        prediction = [None]  # Pas de prédiction
        prediction_proba = np.array([[1 - diabetic_ratio[cluster], diabetic_ratio[cluster]]])
    elif model_type == "DBSCAN":
        df = pd.read_csv('../data/diabetes.csv')
        df_clean = df.copy()
        for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df_clean[column] = df_clean[column].replace(0, np.nan)
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        X = df_clean.drop('Outcome', axis=1)
        X_scaled = scaler.transform(X)
        clusters = model.fit_predict(X_scaled)
        distances = np.sqrt(np.sum((X_scaled - user_input_scaled)**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_cluster = clusters[closest_idx]
        is_outlier = closest_distance > 2.0 or closest_cluster == -1
        df_with_clusters = pd.DataFrame({'cluster': clusters, 'outcome': df_clean['Outcome']})
        if closest_cluster == -1:
            cluster_size = 0
            diabetic_ratio = None
        else:
            cluster_size = (df_with_clusters['cluster'] == closest_cluster).sum()
            diabetic_ratio = df_with_clusters.groupby('cluster')['outcome'].mean()[closest_cluster]
        prediction = [None]
        prediction_proba = np.array([[0, 0]])

# Onglet Prédiction
with tabs[0]:
    # Disposition en deux colonnes
    col1, col2 = st.columns([1, 1])

    # Colonne 1: Résultat de la prédiction
    with col1:
        st.markdown('<div class="section-title">Analyse descriptive du groupe</div>', unsafe_allow_html=True)
        if model_type == "KMeans":
            # Determine risk level based on diabetic ratio in cluster
            diabetic_ratio_val = float(diabetic_ratio[cluster])
            risk_class = "high-risk" if cluster == 0 else "low-risk"  # Cluster 0 is high risk (55%), Cluster 1 is low risk (17%)
            
            st.markdown(f"""
            <div class="result-box {risk_class}">
                <h2>🧩 Groupe assigné : <span style='color:#6A36FC'>Cluster {cluster}</span></h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<b>Taille du groupe :</b> {cluster_size} patients", unsafe_allow_html=True)
            st.markdown(f"<b>Proportion de diabétiques dans ce groupe :</b> {diabetic_ratio_val:.1%}", unsafe_allow_html=True)
            st.markdown("""
            <div class='stInfo'>
            KMeans regroupe les profils similaires. Cette information n'est pas une prédiction médicale mais une description de votre proximité avec d'autres patients.
            </div>
            """, unsafe_allow_html=True)
        elif model_type == "DBSCAN":
            if is_outlier:
                st.markdown(f"""
                <div class="result-box high-risk">
                    <h2>🔎 Profil atypique (outlier)</h2>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class='stInfo'>
                Votre profil est rare ou isolé dans la population étudiée selon DBSCAN. Cela ne constitue pas une prédiction médicale.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box low-risk">
                    <h2>🧩 Groupe DBSCAN : <span style='color:#6A36FC'>Cluster {closest_cluster}</span></h2>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<b>Taille du groupe :</b> {cluster_size} patients", unsafe_allow_html=True)
                if diabetic_ratio is not None:
                    try:
                        diabetic_ratio_val = float(diabetic_ratio)
                    except Exception:
                        diabetic_ratio_val = float(diabetic_ratio)
                    st.markdown(f"<b>Proportion de diabétiques dans ce groupe :</b> {diabetic_ratio_val:.1%}", unsafe_allow_html=True)
                st.markdown("""
                <div class='stInfo'>
                DBSCAN regroupe les profils similaires et détecte les profils atypiques. Cette information n'est pas une prédiction médicale mais une description de votre proximité avec d'autres patients.
                </div>
                """, unsafe_allow_html=True)
        
        # Affichage des probabilités ou mesures de similarité selon le type d'algorithme
        col1a, col1b = st.columns(2)
        
        if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
            # Pour les algorithmes supervisés, montrer les probabilités
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
        else:
            # Pour les algorithmes non supervisés, montrer d'autres métriques
            if model_type == "KMeans":
                # Pour KMeans, montrer à quel cluster appartient le patient
                with col1a:
                    st.metric(
                        label="Groupe assigné",
                        value=f"Cluster {cluster}" if 'cluster' in locals() else "N/A"
                    )
                with col1b:
                    st.metric(
                        label="Proportion de diabétiques dans ce groupe",
                        value=f"{prediction_proba[0][1]:.1%}"
                    )
            else:  # DBSCAN
                # Pour DBSCAN, montrer si c'est un outlier et la mesure de similarité
                with col1a:
                    st.metric(
                        label="Type de profil",
                        value="Atypique" if is_outlier else "Standard"
                    )
        
        # Interprétation des résultats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Interprétation</div>', unsafe_allow_html=True)
        
        if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
            # Interprétation pour les algorithmes supervisés
            if prediction[0] == 1:
                st.markdown("""
                - **Risque élevé**: Le modèle a détecté des facteurs de risque significatifs
                - Cette prédiction est basée sur l'ensemble des caractéristiques
                - Un suivi médical est recommandé pour une évaluation complète
                """)
            else:
                st.markdown("""
                - **Risque faible**: Le modèle n'a pas détecté de facteurs de risque significatifs
                - Maintenir un mode de vie sain et des contrôles médicaux réguliers est recommandé
                - Cette prédiction est indicative et non un diagnostic médical
                """)
        else:
            # Interprétation pour les algorithmes non supervisés
            if model_type == "KMeans":
                if prediction[0] is None:
                    st.markdown("""
                    - **Groupe à risque**: Vous avez été assigné au cluster {cluster} qui contient une proportion élevée de patients diabétiques
                    - Ce regroupement est basé sur des similarités dans les profils de patients
                    - Il s'agit d'une analyse de similarité et non d'un diagnostic
                    """)
                else:
                    st.markdown(f"""
                    - **Groupe à faible risque**: Vous avez été assigné au cluster {cluster} qui contient principalement des patients non-diabétiques
                    - Ce regroupement est basé sur des similarités dans les profils de patients
                    - Maintenir un mode de vie sain est toujours recommandé
                    """)
            else:  # DBSCAN
                if prediction[0] is None:
                    st.markdown("""
                    - **Profil atypique**: Votre profil ne correspond pas clairement à un groupe établi
                    - La comparaison est basée sur la proximité avec des patients diabétiques et non-diabétiques
                    - Une évaluation médicale plus approfondie est recommandée
                    """)
                else:
                    if prediction[0] == 1:
                        st.markdown("""
                        - **Profil similaire aux patients diabétiques**: Votre profil présente des similarités importantes avec des patients diagnostiqués
                        - Cette analyse est basée sur la proximité de vos caractéristiques avec celles de patients diabétiques connus
                        - Un suivi médical est recommandé pour évaluer vos facteurs de risque
                        """)
                    else:
                        st.markdown("""
                        - **Profil similaire aux patients non-diabétiques**: Votre profil présente des caractéristiques proches de personnes sans diabète
                        - Cette analyse est basée sur la similarité de vos données avec celles de patients non-diabétiques
                        - Maintenir un mode de vie sain reste essentiel pour prévenir les risques futurs
                        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonne 2: Visualisation du niveau de risque
    with col2:
        # Only show arc/gauge and bar chart for supervised models
        if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
            st.markdown('<div class="section-title">Niveau de risque</div>', unsafe_allow_html=True)
            # Graphique à barres horizontales
            fig = go.Figure()
            
            if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
                labels = ['Non-diabète', 'Diabète']
            else:
                if model_type == "KMeans":
                    labels = ['Patients non-diabétiques', 'Patients diabétiques']
                else:  # DBSCAN
                    labels = ['Similarité avec profils non-diabétiques', 'Similarité avec profils diabétiques']
            
            fig.add_trace(go.Bar(
                y=labels,
                x=[prediction_proba[0][0], prediction_proba[0][1]],
                orientation='h',
                marker_color=['#10B981', '#EF4444'],
                text=[f"{prediction_proba[0][0]:.1%}", f"{prediction_proba[0][1]:.1%}"],
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis=dict(
                    title='Proportion' if model_type not in ["RandomForest", "LogisticRegression", "SVM"] else 'Probabilité',
                    tickformat='.0%',
                    range=[0, 1.1]
                ),
                margin=dict(l=20, r=40, t=20, b=20),
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Jauge de risque avec titre différent selon le type d'algorithme
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
            
            gauge_title = ""
            if model_type not in ["RandomForest", "LogisticRegression", "SVM"]:
                if model_type == "KMeans":
                    gauge_title = "Proportion dans le groupe"
                else:  # DBSCAN
                    gauge_title = "Indice de similarité"
            
            fig.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=30, b=20),
                title={
                    'text': gauge_title,
                    'y': 0.85,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Légende pour la jauge avec libellés différents selon le type d'algorithme
            if model_type in ["RandomForest", "LogisticRegression", "SVM"]:
                st.markdown("""
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #10B981; font-weight: 600;">Faible (0-30%)</span>
                    <span style="color: #F59E0B; font-weight: 600;">Modéré (30-70%)</span>
                    <span style="color: #EF4444; font-weight: 600;">Élevé (70-100%)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #10B981; font-weight: 600;">Faible similarité</span>
                    <span style="color: #F59E0B; font-weight: 600;">Similarité modérée</span>
                    <span style="color: #EF4444; font-weight: 600;">Forte similarité</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            # For KMeans and DBSCAN, show only a summary card
            pass

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
            color_continuous_scale=[[0, '#87CEEB'], [0.3, '#4682B4'], [0.7, '#1E90FF'], [1, '#000080']]  # Custom blue gradient from light to dark blue
        )
        
        fig.update_layout(
            height=350,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

