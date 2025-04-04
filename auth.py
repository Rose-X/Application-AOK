import streamlit as st

USERS = {
    "admin": "password123",
    "user1": "test123",
    "user2": "demo456"
}

def authenticate():
    st.sidebar.title("Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    login_button = st.sidebar.button("Se connecter")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if login_button:
        if USERS.get(username) == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"Bienvenue, {username} !")
        else:
            st.sidebar.error("Identifiants incorrects.")

    return st.session_state.authenticated
   
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
