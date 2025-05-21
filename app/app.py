import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Diabète",
    page_icon="🏥",
    layout="wide"
)

# Titre de l'application
st.title("🏥 Application de Prédiction du Diabète")

# Charger le modèle (d'abord, vérifions s'il existe, sinon nous allons l'entraîner)
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'
    
    # Si les modèles n'existent pas, on les entraîne
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        # Charger les données
        df = pd.read_csv('../data/diabetes.csv')
        
        # Prétraitement (similaire à celui fait dans le notebook)
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
        
        # Entraînement du modèle (nous utilisons Random Forest)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_scaled, y)
        
        # Sauvegarde du modèle et du scaler
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    # Charger les modèles sauvegardés
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Charger le modèle et le scaler
model, scaler = load_model()

# Création de la sidebar pour l'entrée des données
st.sidebar.header("Entrée des données patient")

# Fonction pour obtenir les entrées utilisateur
def get_user_input():
    pregnancies = st.sidebar.slider("Nombre de grossesses", 0, 17, 3)
    glucose = st.sidebar.slider("Concentration en glucose (mg/dL)", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Pression artérielle (mm Hg)", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Épaisseur de la peau (mm)", 0, 99, 20)
    insulin = st.sidebar.slider("Insuline (µU/ml)", 0, 846, 79)
    bmi = st.sidebar.slider("Indice de masse corporelle (kg/m²)", 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider("Fonction de pedigree du diabète", 0.078, 2.42, 0.3725)
    age = st.sidebar.slider("Âge", 21, 81, 29)
    
    # Stocker les entrées dans un dictionnaire
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
    
    # Convertir le dictionnaire en dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Obtenir les entrées utilisateur
user_input = get_user_input()

# Afficher les caractéristiques du patient
st.subheader("Caractéristiques du patient")
st.write(user_input)

# Standardiser les entrées utilisateur
user_input_scaled = scaler.transform(user_input)

# Prédire et afficher le résultat
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Créer deux colonnes pour l'affichage
col1, col2 = st.columns(2)

# Colonne 1: Résultat de la prédiction
with col1:
    st.subheader("Résultat de la prédiction")
    if prediction[0] == 1:
        st.error("⚠️ Risque élevé de diabète")
    else:
        st.success("✅ Risque faible de diabète")
    
    st.write(f"Probabilité de diabète: {prediction_proba[0][1]:.2%}")
    st.write(f"Probabilité d'absence de diabète: {prediction_proba[0][0]:.2%}")

# Colonne 2: Jauge de risque
with col2:
    st.subheader("Niveau de risque")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(['Risque'], [prediction_proba[0][1]], color='red', alpha=0.6)
    ax.barh(['Risque'], [prediction_proba[0][0]], color='green', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylabel('')
    ax.set_xlabel('Probabilité')
    plt.tight_layout()
    st.pyplot(fig)

# Afficher l'importance des caractéristiques
st.subheader("Importance des caractéristiques")
feature_importance = pd.DataFrame({
    'Feature': user_input.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
plt.tight_layout()
st.pyplot(fig)

# Informations sur le modèle
st.subheader("À propos du modèle")
st.write("""
Cette application utilise un modèle de Random Forest pour prédire le risque de diabète en se basant sur des caractéristiques médicales.
Le modèle a été entraîné sur le dataset Pima Indians Diabetes Database, qui contient des données médicales de femmes d'origine Pima.
""")

# Footer
st.markdown("---")
st.markdown("Créé dans le cadre d'un mini-projet d'analyse et fouille de données.")