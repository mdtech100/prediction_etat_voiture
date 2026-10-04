# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title="Prédiction Voiture — Dakar",
    page_icon="🚗",
    layout="centered"
)

# ── Chargement des artefacts ──────────────────────────────────
@st.cache_resource
def charger_artefacts():
    modele    = joblib.load("gb_model.joblib")
    encoders  = joblib.load("encoders.joblib")
    scaler    = joblib.load("scaler.joblib")
    uniques   = joblib.load("uniques.joblib")
    return modele, encoders, scaler, uniques

modele, encoders, scaler, uniques = charger_artefacts()

# ── En-tête ───────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0'>
        <h1 style='color:#1a73e8; font-size:2.2rem'>🚗 AutoPredict Dakar</h1>
        <p style='color:#555; font-size:1rem'>
            Prédisez si une voiture est <b>Venante</b> ou <b>d'Occasion</b>
            à partir de ses caractéristiques
        </p>
    </div>
    <hr style='margin-bottom:1.5rem'>
""", unsafe_allow_html=True)

# ── Formulaire de saisie ──────────────────────────────────────
st.subheader("📋 Caractéristiques du véhicule")

col1, col2 = st.columns(2)

# Selectbox — indices au lieu de clés
with col1:
    marque = st.selectbox(
        "🏷️ Marque",
        options=sorted(uniques[0]),   # list_marque
        help="Sélectionnez la marque du véhicule"
    )

with col2:
    transmission = st.selectbox(
        "⚙️ Transmission",
        options=sorted(uniques[1]),   # list_transmission
        help="Type de boîte de vitesses"
    )
    quartier = st.selectbox(
        "📍 Quartier",
        options=sorted(uniques[2]),   # list_quartier
        help="Quartier de Dakar où est vendue la voiture"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Prédiction ────────────────────────────────────────────────
if st.button("🔍 Lancer la prédiction", use_container_width=True, type="primary"):

    # Encodage des variables catégorielles
    # encoders = [encoder_marque, encoder_transmission, encoder_quartier, encoder_etat]
    # Ordre : Marque, Transmission, Quartier
    try:
        marque_enc       = encoders[0].transform([marque])[0]
        transmission_enc = encoders[1].transform([transmission])[0]
        quartier_enc     = encoders[2].transform([quartier])[0]
    except ValueError as e:
        st.error(f"⚠️ Valeur inconnue lors de l'encodage : {e}")
        st.stop()

    # Construction du vecteur de features
    # Ordre attendu par le modèle : Marque, Transmission, Quartier, Année, Prix
    X_input = np.array([[
        marque_enc,
        transmission_enc,
        quartier_enc,
        annee,
        prix
    ]])

    # Mise à l'échelle
    X_scaled = scaler.transform(X_input)

    # Prédiction
    prediction    = modele.predict(X_scaled)[0]
    probabilites  = modele.predict_proba(X_scaled)[0]

    prob_occasion = probabilites[0] * 100
    prob_venante  = probabilites[1] * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Résultat de la prédiction")

    # Affichage du résultat principal
    if prediction == 1:
        st.success("✅ Cette voiture est probablement **VENANTE** (importée)")
        couleur = "#1a73e8"
        emoji   = "✈️"
        label   = "VENANTE"
    else:
        st.warning("🔄 Cette voiture est probablement **D'OCCASION** (locale)")
        couleur = "#f4a221"
        emoji   = "🔄"
        label   = "D'OCCASION"

    # Carte de résultat
    st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {couleur}15, {couleur}30);
            border-left: 5px solid {couleur};
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
        '>
            <h2 style='color:{couleur}; margin:0'>
                {emoji} {label}
            </h2>
            <p style='margin:0.5rem 0 0 0; color:#333'>
                Confiance du modèle : <b>{max(prob_occasion, prob_venante):.1f}%</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Probabilités détaillées
    st.markdown("#### Probabilités détaillées")
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            label="🔄 D'Occasion",
            value=f"{prob_occasion:.1f}%"
        )
        st.progress(prob_occasion / 100)

    with col_b:
        st.metric(
            label="✈️ Venante",
            value=f"{prob_venante:.1f}%"
        )
        st.progress(prob_venante / 100)

    # Récapitulatif des données saisies
    st.markdown("#### 📋 Récapitulatif de la saisie")
    recap = pd.DataFrame({
        "Caractéristique": ["Marque", "Transmission", "Quartier", "Année", "Prix"],
        "Valeur": [
            marque,
            transmission,
            quartier,
            str(annee),
            f"{prix:,} FCFA"
        ]
    })
    st.dataframe(recap, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
    <hr>
    <div style='text-align:center; color:#999; font-size:0.85rem; padding:1rem 0'>
        Modèle : <b>Gradient Boosting</b> — Accuracy : <b>82.6%</b><br>
        Données : <a href='https://www.expat-dakar.com/voitures/dakar'
        target='_blank'>Expat-Dakar</a> — Dakar, Sénégal
    </div>
""", unsafe_allow_html=True)