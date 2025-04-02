import streamlit as st
import pandas as pd

@st.cache_data
def charger_donnees():
    """ Charge le fichier CSV contenant les films et les évaluations utilisateurs """
    return pd.read_csv("user_ratings_genres_mov.csv")
def afficher_recommandations(recommendations):
    """ Affiche les recommandations de films """
    st.header("4. Affichage des résultats")
    for index, row in recommendations.iterrows():
        st.subheader(row['title'])
        st.write(f"Genres : {row['genres']}")
        st.write(f"Note : {row['rating']}")
        st.write(f"Année : {row['year']}")
        st.image(row['poster'], width=100)
    if recommendations.empty:
        st.warning("Aucune recommandation trouvée.")