import streamlit as st
import pandas as pd
from test_r import gerar_regras_com_r_via_subprocess

st.set_page_config(page_title="Gerador de Regras com R", layout="wide")
st.title("Gerador de Regras de Associação com R (arules)")

# --- Upload de CSV ou DataFrame de exemplo ---
st.sidebar.header("Opções de Entrada")
upload = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

if upload is not None:
    df = pd.read_csv(upload, dtype=str)
else:
    st.info("Nenhum CSV enviado. Usando DataFrame de exemplo.")
    data = {
        "A": ["sim", "não", "sim", "sim", "não", "sim"],
        "B": ["x", "y", "x", "y", "x", "y"],
        "C": ["verde", "verde", "azul", "azul", "verde", "azul"]
    }
    df = pd.DataFrame(data)

st.subheader("Dados de Entrada")
st.dataframe(df)

# --- Configuração de parâmetros ---
st.sidebar.header("Parâmetros Apriori")
supp = st.sidebar.number_input("Suporte mínimo", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
conf = st.sidebar.number_input("Confiança mínima", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

# --- Botão para gerar regras ---
if st.button("Gerar Regras"):
    try:
        with st.spinner("Gerando regras..."):
            # Chamando função Python que roda o R
            df_regras = gerar_regras_com_r_via_subprocess(df, script_path="gerar_regras.R")
        
        if df_regras.empty:
            st.warning("Nenhuma regra foi gerada com os parâmetros escolhidos.")
        else:
            st.success(f"{len(df_regras)} regra(s) gerada(s) com sucesso!")
            st.subheader("Regras geradas")
            st.dataframe(df_regras)
    except Exception as e:
        st.error(f"Erro ao gerar regras: {e}")
