import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Chargement des données
@st.cache_data
def charger_donnees():
    return pd.read_csv("user_ratings_genres_mov.csv")

df = charger_donnees()

st.title("Application de Recommandation de Films !!!!!!!")

# 1. Création du profil utilisateur
st.header("1. Création de votre profil utilisateur")
profil_utilisateur = []

films_disponibles = df['title'].unique()

for i in range(1, 4):
    with st.expander(f"Film {i}"):
        film = st.selectbox(f"Sélectionnez un film {i}", options=films_disponibles, key=f"film_{i}")
        note = st.slider(f"Note (sur 5)", 1, 5, key=f"note_{i}")
        genres = st.multiselect("Genres", df['genres'].str.split('|').explode().unique(), key=f"genres_{i}")

        if film and genres:
            profil_utilisateur.append({"film": film, "note": note, "genres": genres})

if st.button("Valider le profil"):
    if len(profil_utilisateur) == 3:
        st.success("Profil enregistré.")
    else:
        st.warning("Veuillez saisir 3 films avec leurs genres.")

# 2. Implémentation des approches de recommandation
st.header("2. Implémentation des approches de recommandation")
approche = st.radio("Méthode :", ["Collaborative - Mémoire", "Collaborative - Modèle KNN", "Basée sur le contenu"])

# 3. Calcul des résultats
if st.button("Calculer les résultats") and len(profil_utilisateur) == 3:
    st.header("3. Résultats")

    utilisateur_df = pd.DataFrame(profil_utilisateur)

    if approche == "Basée sur le contenu":
        film_prefere = utilisateur_df.sort_values("note", ascending=False).iloc[0]
        genres_pref = set(film_prefere["genres"])

        def jaccard_score(genres_film):
            set_film = set(genres_film.split('|'))
            intersection = len(set_film & genres_pref)
            union = len(set_film | genres_pref)
            return intersection / union if union != 0 else 0

        df["similarite"] = df["genres"].apply(jaccard_score)
        recommendations = df.sort_values("similarite", ascending=False).head(5)
        st.header("4. Affichage des résultats")
        st.write("Méthode : basée sur le contenu")
        st.dataframe(recommendations[["title", "similarite"]].rename(columns={"title": "Titre", "similarite": "Score de similarité"}))

    elif approche == "Collaborative - Mémoire":
        ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating')
        user_profile = utilisateur_df.set_index("film")["note"]
        ratings_matrix.loc['nouvel_utilisateur'] = user_profile
        ratings_matrix_filled = ratings_matrix.fillna(0)

        similarity = cosine_similarity(ratings_matrix_filled)
        similarity_df = pd.DataFrame(similarity, index=ratings_matrix_filled.index, columns=ratings_matrix_filled.index)

        similar_users = similarity_df["nouvel_utilisateur"].drop("nouvel_utilisateur").sort_values(ascending=False).head(3)
        scores = ratings_matrix_filled.loc[similar_users.index].T.dot(similar_users)
        scores = scores / similar_users.sum()

        scores = scores.drop(user_profile.index, errors='ignore')
        recommandations = scores.sort_values(ascending=False).head(5)

        st.header("4. Affichage des résultats")
        st.write("Méthode : collaborative basée sur la mémoire")
        st.dataframe(recommandations.reset_index().rename(columns={0: "Score", "title": "Titre"}))

    elif approche == "Collaborative - Modèle KNN":
        ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(ratings_matrix)

        user_profile = utilisateur_df.set_index("film")["note"]
        user_vector = pd.Series(0, index=ratings_matrix.columns)
        for film, note in user_profile.items():
            if film in user_vector:
                user_vector[film] = note

        distances, indices = model.kneighbors([user_vector], n_neighbors=3)
        voisins = ratings_matrix.iloc[indices[0]]
        moyennes = voisins.mean().drop(user_profile.index, errors='ignore')
        recommandations = moyennes.sort_values(ascending=False).head(5)

        st.header("4. Affichage des résultats")
        st.write("Méthode : collaborative basée sur un modèle KNN")
        st.dataframe(recommandations.reset_index().rename(columns={0: "Score", "title": "Titre"}))
