import streamlit as st
import plotly.express as px
import pandas as pd
import time

st.set_page_config(page_title="Teste de Abas de Análises", layout="wide")

# Inicializa a lista de análises salvas
if "analises_salvas" not in st.session_state:
    st.session_state.analises_salvas = []

st.title("🧠 Teste de Abas Dinâmicas de Análises")

# Simulação de geração de uma nova análise
st.subheader("Gerar nova análise")
parametro = st.number_input("Parâmetro de exemplo", 0.0, 1.0, 0.5, step=0.1)

if st.button("Gerar análise"):
    inicio = time.time()
    df = pd.DataFrame({
        "x": range(10),
        "y": [parametro * i for i in range(10)]
    })
    fig = px.line(df, x="x", y="y", title=f"Análise com parâmetro {parametro}")
    tempo_exec = time.time() - inicio

    nova_analise = {
        "nome": f"Análise {len(st.session_state.analises_salvas) + 1}",
        "parametro": parametro,
        "tempo": tempo_exec,
        "grafico": fig
    }
    st.session_state.analises_salvas.append(nova_analise)
    st.success("✅ Nova análise salva com sucesso!")

# Exibe as abas existentes
if st.session_state.analises_salvas:
    nomes_abas = [a["nome"] for a in st.session_state.analises_salvas]
    abas = st.tabs(nomes_abas)

    for i, a in enumerate(st.session_state.analises_salvas):
        with abas[i]:
            st.write(f"**Parâmetro:** {a['parametro']}")
            st.write(f"**Tempo de execução:** {a['tempo']:.2f}s")
            st.plotly_chart(a["grafico"], use_container_width=True, key=f"grafico_{i}")

