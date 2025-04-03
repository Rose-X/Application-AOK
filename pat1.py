import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from auth import authenticate, logout
from data_loader import charger_donnees
import streamlit as st

st.markdown(
    """
    <style>
    .st-emotion-cache-bm2z3a  {
        background-color: #003366; /* Remplace cette couleur par celle de ton choix */
    }
    .st-emotion-cache-6qob1r  {
        background-color: #002244 !important; /* Remplace cette couleur par celle de ton choix */
        }
    .st-emotion-cache-102y9h7 {
        color: white !important;
    }
    .st-emotion-cache-y73bov {
        color: white !important;
    }
    .st-emotion-cache-1s2v671 {
        color: white !important;
    }
    .st-emotion-cache-1fmytai {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialisation de la session pour gérer la connexion
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Vérification de l'authentification
if not st.session_state.authenticated:
    if authenticate():
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.warning("Veuillez vous connecter pour accéder à l'application.")
        st.stop()

# Bouton de déconnexion
if st.sidebar.button("🔒 Se déconnecter"):
    logout()
    st.session_state.authenticated = False
    st.rerun()

# Chargement des données
df = charger_donnees()

# Barre de navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Accueil", "Tous les films", "À propos"])

if page == "Accueil":
    st.title("Application de Recommandation de Films")

    # 1. Création du profil utilisateur
    st.header("Création de votre profil utilisateur")
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
    st.header("Recommandation")
    approche = st.radio("Méthode :", ["Collaborative - Mémoire", "Collaborative - Modèle KNN", "Basée sur le contenu"])

    # 3. Calcul des résultats
    if st.button("Chercher les films") and len(profil_utilisateur) == 3:

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
            st.header("Les films recommandés par le contenu")
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

            st.header("Les films collaboratifs memoires recommandés")
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

            st.header("Les films collaboratifs KNN recommandés")
            st.write("Méthode : collaborative basée sur un modèle KNN")
            st.dataframe(recommandations.reset_index().rename(columns={0: "Score", "title": "Titre"}))

elif page == "Tous les films":
    st.title(" Tous les Films")

    # Champ de recherche
    search = st.text_input("🔍 Rechercher un film :", "")

    # Liste des films avec filtrage
    filtered_df = df[df["title"].str.contains(search, case=False, na=False)]

    # Affichage des films
    st.dataframe(filtered_df[["title", "genres", "rating"]].rename(columns={
        "title": "Titre",
        "genres": "Genres",
        "rating": "Note Moyenne"
    }))

elif page == "À propos":
    st.title("📊 À propos de l'application")
    st.write("""
    Cette application de recommandation de films utilise des approches collaboratives et basées sur le contenu.
    Elle permet aux utilisateurs de créer leur profil et d'obtenir des recommandations en fonction de leurs goûts cinématographiques.
    """)
    st.write("Développée par Aiden et Alexandre - 2023")
