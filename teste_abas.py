from streamlit_option_menu import option_menu
import streamlit as st

st.set_page_config(page_title="Sidebar estilizada", layout="wide")

with st.sidebar:
    escolha = option_menu(
        "Navegação",
        ["Aba 1", "Aba 2", "Aba 3", "Aba 4"],
        icons=["house", "graph-up", "gear", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0366d6"},
        },
    )

st.header(f"Conteúdo da {escolha}")
for i in range(100):
    st.write(f"Linha {i+1} — {escolha}")