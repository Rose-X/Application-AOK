# Application-AOK


Voici une explication **très détaillée** du code, en passant en revue chaque paramètre et chaque étape, afin que tu puisses bien comprendre et expliquer l’ensemble du fonctionnement de l’application.

---

## 1. Importation des modules

```python
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from auth import authenticate, logout
from data_loader import charger_donnees
import streamlit as st
```

- **`import streamlit as st`**  
  - **Rôle :** Importer le module Streamlit qui permet de créer des interfaces web interactives pour des applications Python.  
  - **Alias :** On utilise `st` pour raccourcir l'appel des fonctions (exemple : `st.markdown`, `st.sidebar`).

- **`import pandas as pd`**  
  - **Rôle :** Importer la bibliothèque Pandas pour manipuler et analyser des données sous forme de tableaux (DataFrame).  
  - **Alias :** `pd` est un raccourci standard pour Pandas.

- **`from sklearn.metrics.pairwise import cosine_similarity`**  
  - **Rôle :** Importer la fonction `cosine_similarity` qui calcule la similarité cosinus entre des vecteurs.  
  - **Utilisation :** Utile pour mesurer la ressemblance entre des utilisateurs ou des films dans l’approche collaborative basée sur la mémoire.

- **`from sklearn.neighbors import NearestNeighbors`**  
  - **Rôle :** Importer la classe `NearestNeighbors` pour créer un modèle KNN (k plus proches voisins).  
  - **Paramètres importants :** On pourra préciser la métrique de distance (ici la similarité cosinus) et l’algorithme de calcul (ici « brute » pour un calcul exhaustif).

- **`import numpy as np`**  
  - **Rôle :** Importer la bibliothèque NumPy pour effectuer des opérations mathématiques et manipuler des tableaux numériques.

- **`from auth import authenticate, logout`**  
  - **Rôle :** Importer deux fonctions personnalisées :
    - `authenticate()`: pour vérifier si un utilisateur est bien connecté.
    - `logout()`: pour gérer la déconnexion de l’utilisateur.

- **`from data_loader import charger_donnees`**  
  - **Rôle :** Importer la fonction `charger_donnees()` qui permet de charger les données des films (probablement depuis un fichier ou une base de données).

- **`import streamlit as st`**  
  - **Remarque :** Cette ligne est redondante, car Streamlit a déjà été importé. Elle n’a pas d’impact fonctionnel, mais peut être nettoyée.

---

## 2. Personnalisation du style avec CSS

```python
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
```

- **`st.markdown()`**  
  - **Rôle :** Permet d’afficher du contenu au format Markdown ou HTML.  
  - **Paramètre `unsafe_allow_html=True` :** Nécessaire pour autoriser l’injection de code HTML/CSS. Attention à la sécurité, mais ici c’est pour personnaliser l’interface.

- **Code CSS inclus :**
  - Les classes comme `.st-emotion-cache-bm2z3a` sont générées automatiquement par Streamlit pour certains composants.  
  - **`background-color` et `color`** : Ces propriétés définissent respectivement la couleur de fond et la couleur du texte.  
  - Le mot-clé `!important` force l’application de la règle même si d’autres styles existent.

---

## 3. Initialisation et gestion de la session pour l’authentification

### a. Initialisation de la variable de session

```python
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
```

- **`st.session_state` :**  
  - Un dictionnaire persistant utilisé pour garder en mémoire des variables durant la session de l’utilisateur.  
  - Ici, on vérifie si la clé `"authenticated"` existe. Sinon, on la crée et on la fixe à `False` pour indiquer que l’utilisateur n’est pas encore connecté.

### b. Vérification et authentification

```python
if not st.session_state.authenticated:
    if authenticate():
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.warning("Veuillez vous connecter pour accéder à l'application.")
        st.stop()
```

- **Condition :** Si l’utilisateur n’est pas authentifié (`not st.session_state.authenticated`).
- **`authenticate()` :**  
  - Fonction qui demande les identifiants de connexion et retourne `True` si la connexion est réussie.
- **`st.session_state.authenticated = True` :**  
  - Mise à jour de l’état de connexion.
- **`st.rerun()` :**  
  - Recharge l’application pour appliquer l’état modifié (l’utilisateur connecté peut maintenant accéder à l’application).
- **`st.warning()` et `st.stop()` :**  
  - Si l’authentification échoue, affiche un message d’avertissement et arrête l’exécution du script afin d’empêcher l’accès.

### c. Bouton de déconnexion

```python
if st.sidebar.button("🔒 Se déconnecter"):
    logout()
    st.session_state.authenticated = False
    st.rerun()
```

- **`st.sidebar.button("🔒 Se déconnecter")` :**  
  - Crée un bouton dans la barre latérale pour permettre à l’utilisateur de se déconnecter.
- **`logout()` :**  
  - Fonction qui effectue les opérations de déconnexion (par exemple, nettoyage des cookies ou jetons de session).
- **Réinitialisation de l’état :**  
  - On remet `"authenticated"` à `False` et on recharge l’application avec `st.rerun()`.

---

## 4. Chargement des données

```python
df = charger_donnees()
```

- **`charger_donnees()` :**  
  - Fonction personnalisée qui charge et retourne un DataFrame contenant les films, les notes, les genres et possiblement d’autres informations (comme `userId`).  
  - Le DataFrame `df` sera utilisé pour générer les recommandations et afficher la liste des films.

---

## 5. Barre de navigation

```python
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Accueil", "Tous les films", "À propos"])
```

- **`st.sidebar.title()` :**  
  - Affiche un titre dans la barre latérale.
- **`st.sidebar.radio()` :**  
  - Crée un bouton radio permettant de choisir entre différentes pages (ici, "Accueil", "Tous les films" et "À propos").  
  - La variable `page` stocke le choix de l’utilisateur.

---

## 6. Page "Accueil" – Création du profil utilisateur et recommandation

### a. Titre et en-tête

```python
if page == "Accueil":
    st.title("Application de Recommandation de Films")
    st.header("Création de votre profil utilisateur")
```

- **`st.title()` et `st.header()` :**  
  - Affichent respectivement un titre principal et une en-tête pour la section de création du profil.

### b. Initialisation du profil et récupération des films disponibles

```python
profil_utilisateur = []
films_disponibles = df['title'].unique()
```

- **`profil_utilisateur` :**  
  - Liste qui va stocker les choix de films, leurs notes et genres pour l’utilisateur.
- **`films_disponibles` :**  
  - Extraction de la liste unique des titres de films à partir du DataFrame `df` pour alimenter le menu de sélection.

### c. Boucle de sélection des films (3 films)

```python
for i in range(1, 4):
    with st.expander(f"Film {i}"):
        film = st.selectbox(f"Sélectionnez un film {i}", options=films_disponibles, key=f"film_{i}")
        note = st.slider(f"Note (sur 5)", 1, 5, key=f"note_{i}")
        genres = st.multiselect("Genres", df['genres'].str.split('|').explode().unique(), key=f"genres_{i}")

        if film and genres:
            profil_utilisateur.append({"film": film, "note": note, "genres": genres})
```

- **`for i in range(1, 4)` :**  
  - Itération sur 3 films (de 1 à 3) pour que l’utilisateur saisisse 3 choix.
  
- **`with st.expander(f"Film {i}")` :**  
  - Crée une section repliable pour chaque film afin de ne pas surcharger l’écran.

- **`st.selectbox(...)` :**  
  - Permet de sélectionner un film parmi ceux disponibles.
  - **Paramètres importants :**  
    - `options=films_disponibles` : liste des films à choisir.
    - `key=f"film_{i}"` : clé unique pour différencier chaque composant dans la session.

- **`st.slider(...)` :**  
  - Permet à l’utilisateur de donner une note au film.
  - **Paramètres :**  
    - Valeurs minimum et maximum (1 à 5).
    - `key=f"note_{i}"` pour garder une trace unique de ce slider.

- **`st.multiselect(...)` :**  
  - Permet de sélectionner plusieurs genres pour le film.
  - **Paramètres :**  
    - `df['genres'].str.split('|').explode().unique()` :  
      - **Explication :**  
        - `df['genres'].str.split('|')` : Sépare la chaîne de caractères des genres en une liste (exemple : "Action|Thriller" devient ["Action", "Thriller"]).
        - `.explode()` : Transforme la série de listes en plusieurs lignes, une par genre.
        - `.unique()` : Récupère les genres uniques.
    - `key=f"genres_{i}"` pour l’unicité du composant.

- **Condition `if film and genres:`**  
  - Vérifie que l’utilisateur a bien sélectionné un film ET au moins un genre.
  - **`profil_utilisateur.append({...})` :**  
    - Ajoute un dictionnaire contenant le film choisi, la note et les genres dans la liste `profil_utilisateur`.

### d. Validation du profil utilisateur

```python
if st.button("Valider le profil"):
    if len(profil_utilisateur) == 3:
        st.success("Profil enregistré.")
    else:
        st.warning("Veuillez saisir 3 films avec leurs genres.")
```

- **`st.button("Valider le profil")` :**  
  - Crée un bouton pour confirmer la saisie du profil.
- **Validation :**  
  - Vérifie que la liste contient bien 3 films.
  - Affiche un message de succès ou un avertissement en fonction du résultat.

### e. Sélection de la méthode de recommandation

```python
st.header("Recommandation")
approche = st.radio("Méthode :", ["Collaborative - Mémoire", "Collaborative - Modèle KNN", "Basée sur le contenu"])
```

- **`st.radio()` :**  
  - Permet à l’utilisateur de choisir entre plusieurs approches de recommandation.
  - **Options :**  
    - **Collaborative - Mémoire :** Utilise la similarité cosinus entre utilisateurs.
    - **Collaborative - Modèle KNN :** Utilise un modèle k-plus proches voisins.
    - **Basée sur le contenu :** Compare les genres des films.

### f. Calcul et affichage des recommandations

```python
if st.button("Chercher les films") and len(profil_utilisateur) == 3:
    utilisateur_df = pd.DataFrame(profil_utilisateur)
```

- **Condition :**  
  - Lancer le calcul des recommandations uniquement si l’utilisateur clique sur le bouton et que son profil contient bien 3 films.
- **Conversion en DataFrame :**  
  - Transforme la liste `profil_utilisateur` en un DataFrame pour faciliter les manipulations (tri, sélection, etc.).

#### i. Approche "Basée sur le contenu"

```python
film_prefere = utilisateur_df.sort_values("note", ascending=False).iloc[0]
genres_pref = set(film_prefere["genres"])
```

- **`sort_values("note", ascending=False)` :**  
  - Trie les films du profil de l’utilisateur par note décroissante.
- **`iloc[0]` :**  
  - Sélectionne le film ayant la note la plus élevée (film préféré).
- **Conversion en set :**  
  - `set(film_prefere["genres"])` : Convertit la liste des genres en ensemble pour faciliter la comparaison.

Définition d’une fonction pour le score de similarité (Jaccard) :

```python
def jaccard_score(genres_film):
    set_film = set(genres_film.split('|'))
    intersection = len(set_film & genres_pref)
    union = len(set_film | genres_pref)
    return intersection / union if union != 0 else 0
```

- **But :**  
  - Calculer le score de similarité entre les genres du film préféré et ceux d’un autre film.
- **Détails :**  
  - `genres_film.split('|')` : Sépare la chaîne des genres d’un film.
  - `set_film & genres_pref` : Intersection des ensembles (genres communs).
  - `set_film | genres_pref` : Union des ensembles (tous les genres).
  - Renvoie le score de Jaccard (rapport intersection/union).

Application du score à chaque film :

```python
df["similarite"] = df["genres"].apply(jaccard_score)
recommendations = df.sort_values("similarite", ascending=False).head(5)
```

- **`df["genres"].apply(jaccard_score)` :**  
  - Applique la fonction `jaccard_score` à chaque film de la colonne `genres`.
- **`sort_values("similarite", ascending=False)` :**  
  - Trie les films par score de similarité décroissant.
- **`head(5)` :**  
  - Sélectionne les 5 films avec le score le plus élevé.

Affichage du résultat :

```python
st.header("Les films recommandés par le contenu")
st.write("Méthode : basée sur le contenu")
st.dataframe(recommendations[["title", "similarite"]].rename(columns={"title": "Titre", "similarite": "Score de similarité"}))
```

- **Renommage des colonnes** pour une présentation plus claire.

#### ii. Approche "Collaborative - Mémoire"

```python
ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating')
```

- **`pivot_table` :**  
  - Crée une matrice où chaque ligne représente un utilisateur (`userId`) et chaque colonne un film (`title`).  
  - **`values='rating'`** indique que les valeurs dans la matrice sont les notes données aux films.

```python
user_profile = utilisateur_df.set_index("film")["note"]
ratings_matrix.loc['nouvel_utilisateur'] = user_profile
```

- **`set_index("film")`** :  
  - Transforme le DataFrame du profil utilisateur en une série indexée par le nom du film.
- **Ajout de la ligne 'nouvel_utilisateur'** :  
  - On insère le profil de l’utilisateur dans la matrice des notes.

```python
ratings_matrix_filled = ratings_matrix.fillna(0)
```

- **`fillna(0)` :**  
  - Remplace les valeurs manquantes par 0 pour éviter des problèmes lors du calcul de la similarité.

Calcul de la similarité entre utilisateurs :

```python
similarity = cosine_similarity(ratings_matrix_filled)
similarity_df = pd.DataFrame(similarity, index=ratings_matrix_filled.index, columns=ratings_matrix_filled.index)
```

- **`cosine_similarity` :**  
  - Calcule la similarité cosinus entre les vecteurs de notes de chaque utilisateur.
- **Transformation en DataFrame** :  
  - Permet d’avoir une matrice avec les identifiants des utilisateurs comme index et colonnes.

Sélection des utilisateurs les plus similaires :

```python
similar_users = similarity_df["nouvel_utilisateur"].drop("nouvel_utilisateur").sort_values(ascending=False).head(3)
```

- **`drop("nouvel_utilisateur")` :**  
  - Exclut le nouvel utilisateur de la comparaison.
- **`sort_values(ascending=False)` :**  
  - Trie les utilisateurs par ordre décroissant de similarité.
- **`head(3)` :**  
  - Sélectionne les 3 utilisateurs les plus proches.

Calcul du score pour chaque film non noté par le nouvel utilisateur :

```python
scores = ratings_matrix_filled.loc[similar_users.index].T.dot(similar_users)
scores = scores / similar_users.sum()
scores = scores.drop(user_profile.index, errors='ignore')
recommandations = scores.sort_values(ascending=False).head(5)
```

- **`ratings_matrix_filled.loc[similar_users.index]` :**  
  - Récupère les notes des utilisateurs similaires.
- **`.T.dot(similar_users)` :**  
  - Effectue un produit scalaire pondéré par les scores de similarité pour obtenir un score global pour chaque film.
- **Normalisation** par la somme des similarités pour avoir une moyenne pondérée.
- **`drop(user_profile.index)`** :  
  - Élimine les films déjà notés par le nouvel utilisateur.
- **`sort_values` et `head(5)`** :  
  - Trie les scores et sélectionne les 5 meilleurs films.

Affichage du résultat :

```python
st.header("Les films collaboratifs memoires recommandés")
st.write("Méthode : collaborative basée sur la mémoire")
st.dataframe(recommandations.reset_index().rename(columns={0: "Score", "title": "Titre"}))
```

- **`reset_index()`** :  
  - Permet de remettre l’index sous forme de colonne pour l’affichage.
- **Renommage des colonnes** pour une meilleure lisibilité.

#### iii. Approche "Collaborative - Modèle KNN"

Création de la matrice des notes :

```python
ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
```

- Même principe que précédemment, avec le remplissage des valeurs manquantes par 0.

Création et entraînement du modèle KNN :

```python
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(ratings_matrix)
```

- **Paramètre `metric='cosine'` :**  
  - Indique que la distance sera mesurée par la similarité cosinus.
- **Paramètre `algorithm='brute'` :**  
  - Utilise une recherche exhaustive (brute force) pour trouver les voisins.
- **`model.fit(ratings_matrix)`** :  
  - Entraîne le modèle sur la matrice des notes.

Préparation du vecteur de l’utilisateur :

```python
user_profile = utilisateur_df.set_index("film")["note"]
user_vector = pd.Series(0, index=ratings_matrix.columns)
for film, note in user_profile.items():
    if film in user_vector:
        user_vector[film] = note
```

- **Création de `user_vector` :**  
  - Une série initialisée avec 0 pour tous les films présents dans la matrice.
- **Boucle `for` :**  
  - Met à jour `user_vector` pour les films notés par l’utilisateur avec la note correspondante.

Recherche des voisins les plus proches :

```python
distances, indices = model.kneighbors([user_vector], n_neighbors=3)
```

- **`model.kneighbors()` :**  
  - Renvoie deux tableaux :  
    - `distances` : les distances (ou similarités) entre l’utilisateur et ses voisins.
    - `indices` : les indices des utilisateurs voisins dans la matrice.
  - **Paramètre `n_neighbors=3`** :  
    - On recherche les 3 voisins les plus proches.

Sélection et calcul des moyennes des notes :

```python
voisins = ratings_matrix.iloc[indices[0]]
moyennes = voisins.mean().drop(user_profile.index, errors='ignore')
recommandations = moyennes.sort_values(ascending=False).head(5)
```

- **`ratings_matrix.iloc[indices[0]]` :**  
  - Récupère les notes des utilisateurs voisins.
- **`voisins.mean()` :**  
  - Calcule la moyenne des notes pour chaque film parmi les voisins.
- **`drop(user_profile.index)`** :  
  - Supprime les films que l’utilisateur a déjà notés.
- **Tri et sélection** des 5 films les mieux notés par les voisins.

Affichage des recommandations :

```python
st.header("Les films collaboratifs KNN recommandés")
st.write("Méthode : collaborative basée sur un modèle KNN")
st.dataframe(recommandations.reset_index().rename(columns={0: "Score", "title": "Titre"}))
```

- **Affichage via `st.dataframe`** avec remise à zéro de l’index et renommage des colonnes pour la clarté.

---

## 7. Page "Tous les films" – Recherche et affichage de la liste complète

```python
elif page == "Tous les films":
    st.title(" Tous les Films")
    search = st.text_input("🔍 Rechercher un film :", "")
```

- **`st.text_input` :**  
  - Crée un champ de saisie pour rechercher un film par son titre.
  - **Paramètre `""`** : Valeur par défaut vide.

Filtrage et affichage :

```python
filtered_df = df[df["title"].str.contains(search, case=False, na=False)]
st.dataframe(filtered_df[["title", "genres", "rating"]].rename(columns={
    "title": "Titre",
    "genres": "Genres",
    "rating": "Note Moyenne"
}))
```

- **`df["title"].str.contains(search, case=False, na=False)` :**  
  - Filtre le DataFrame `df` pour ne garder que les films dont le titre contient le texte recherché.
  - **`case=False`** : La recherche n’est pas sensible à la casse.
  - **`na=False`** : En cas de valeur manquante, considère que ce n’est pas une correspondance.
- **`st.dataframe()`** :  
  - Affiche le tableau filtré, avec renommage des colonnes pour une meilleure compréhension.

---

## 8. Page "À propos" – Informations sur l’application

```python
elif page == "À propos":
    st.title("📊 À propos de l'application")
    st.write("""
    Cette application de recommandation de films utilise des approches collaboratives et basées sur le contenu.
    Elle permet aux utilisateurs de créer leur profil et d'obtenir des recommandations en fonction de leurs goûts cinématographiques.
    """)
    st.write("Développée par Aiden et Alexandre - 2023")
```

- **Affichage d’un titre et d’un texte explicatif :**
  - **`st.write()`** : Permet d’afficher du texte formaté.
  - Le texte décrit le fonctionnement général de l’application et mentionne ses développeurs.

---

## Conclusion

Chaque section de ce code a un rôle précis :  
- **Importations** pour disposer des outils nécessaires.  
- **CSS personnalisé** pour adapter l’apparence de l’application.  
- **Gestion de la session** pour contrôler l’accès via l’authentification.  
- **Chargement et préparation des données** pour manipuler la liste des films.  
- **Navigation et pages** pour séparer la création du profil, les recommandations, et l’affichage général.  
- **Différentes méthodes de recommandation** illustrant des approches collaboratives (mémoire et KNN) et basées sur le contenu.  

Chacune des fonctions et paramètres est choisie pour répondre à un besoin spécifique de l’application, allant de la gestion des interactions utilisateur à l’analyse des données pour produire des recommandations pertinentes. Ce niveau de détail te permettra d’expliquer le code point par point lors de tes présentations ou examens.
