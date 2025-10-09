# teste_radio_tabs_style.py
import streamlit as st

st.set_page_config(layout="wide", page_title="Radio como Tabs — Estilizado")

# CSS: sticky + estilo de "abas" para o radio
st.markdown("""
<style>
/* Sticky no radio (menu horizontal) */
div[data-testid="stHorizontalBlock"] > div[role="radiogroup"],
div[role="radiogroup"] {
  position: -webkit-sticky !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 9999 !important;
  background: #ffffff !important;
  padding: 0.3rem 0 !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
}

/* Estilo "abas" para os labels do radio */
div[role="radiogroup"] > label {
  display: inline-block;
  padding: 8px 14px;
  margin-right: 6px;
  border-radius: 8px 8px 0 0;
  border: 1px solid transparent;
  border-bottom: none;
  font-weight: 600;
  cursor: pointer;
}

/* label selecionado — Streamlit costuma adicionar aria-checked="true" no label */
div[role="radiogroup"] > label[aria-checked="true"] {
  background: #ffffff;
  border-color: #ddd;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* espaçamento para o conteúdo não ficar por baixo das abas */
div[data-testid="stHorizontalBlock"] + div,
div[role="radiogroup"] + div {
  margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Teste: Radio estilizado como Abas (Sticky)")

options = ["Aba 1", "Aba 2", "Aba 3", "Aba 4"]
sel = st.radio("", options=options, horizontal=True, key="my_tabs")

if sel == "Aba 1":
    st.header("Conteúdo Aba 1")
    for i in range(100):
        st.write(f"Linha {i+1}")
elif sel == "Aba 2":
    st.header("Conteúdo Aba 2")
    for i in range(100):
        st.write(f"Linha {i+1}")
elif sel == "Aba 3":
    st.header("Conteúdo Aba 3")
    for i in range(100):
        st.write(f"Linha {i+1}")
else:
    st.header("Conteúdo Aba 4")
    for i in range(100):
        st.write(f"Linha {i+1}")
