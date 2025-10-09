import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Teste de Abas Fixas", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
/* força overflow visible quase geral — uso de diagnóstico apenas */
html, body, main, * {
  overflow: visible !important;
  transform: none !important;
}

/* fixa as abas */
div[data-testid="stTabs"] div[data-baseweb="tab-list"],
div[data-testid="stTabs"] div[role="tablist"] {
  position: sticky !important;
  top: 0 !important;
  z-index: 99999 !important;
  background: #fff !important;
  padding: 0.4rem 0 !important;
  box-shadow: 0 3px 8px rgba(0,0,0,0.1) !important;
}

/* espaço para o conteúdo */
div[data-testid="stTabs"] + div { margin-top: 4rem !important; }
</style>
""", unsafe_allow_html=True)

# ===== Abas =====
abas = ["Aba 1", "Aba 2", "Aba 3", "Aba 4"]
# --- coloque as abas dentro de um container com key ---
with st.container(key="top_tabs"):
    tab = st.tabs(abas)

# ===== Conteúdo longo pra rolar =====
with tab[0]:
    st.title("Conteúdo da Aba 1")
    for i in range(100):
        st.write(f"Linha {i+1}")

with tab[1]:
    st.title("Conteúdo da Aba 2")
    for i in range(100):
        st.write(f"Linha {i+1}")

with tab[2]:
    st.title("Conteúdo da Aba 3")
    for i in range(100):
        st.write(f"Linha {i+1}")

with tab[3]:
    st.title("Conteúdo da Aba 4")
    for i in range(100):
        st.write(f"Linha {i+1}")
