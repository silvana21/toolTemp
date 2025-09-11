# main.py
import streamlit as st
import pandas as pd
import analysis
from datetime import datetime, timedelta

st.set_page_config(page_title="Análise Temporal", layout="wide")
st.title("Ferramenta de Análise Temporal de Pull Requests")

# Upload do CSV
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Resumo da base")
    st.write(f"Total de registros na base: {len(df)}")
    st.dataframe(df.head())

    # Detectar colunas de data
    datetime_cols = []
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().mean() > 0.7:
                datetime_cols.append(col)
        except Exception:
            continue

    if not datetime_cols:
        st.error("Nenhuma coluna de data detectada.")
    else:
        st.subheader("Colunas de data detectadas")
        selected_date_col = st.selectbox("Selecione a coluna de data para particionamento", datetime_cols)
        df_data_col = pd.to_datetime(df[selected_date_col], errors='coerce')

        # Base sem a coluna de data para gerar regras
        df_sem_data = df.drop(columns=[selected_date_col])

        st.subheader("Colunas usadas para mineração")
        st.write(df_sem_data.columns.tolist())

        # Parâmetros de mineração
        st.subheader("Configuração do Apriori")
        min_support = st.number_input("Suporte mínimo (%)", min_value=0.0, max_value=100.0, value=1.0)/100.0
        min_confidence = st.number_input("Confiança mínima (%)", min_value=0.0, max_value=100.0, value=50.0)/100.0

        # Guardar regras em session_state
        if "df_regras" not in st.session_state:
            st.session_state.df_regras = None
        if "bases_periodos" not in st.session_state:
            st.session_state.bases_periodos = []

        if st.button("Gerar regras na base completa"):
            st.session_state.df_regras = analysis.gerar_regras_com_mlxtend(df_sem_data, sup=min_support, conf=min_confidence)

        df_regras = st.session_state.df_regras

        if df_regras is not None and not df_regras.empty:
            # Filtrar regras 1 antecedente x 1 consequente
            df_filtrado = df_regras[
                (df_regras['antecedente'].apply(lambda x: len(x.split(','))==1)) &
                (df_regras['consequente'].apply(lambda x: len(x.split(','))==1))
            ]

            # Top 10 por lift
            top10_regras = df_filtrado.sort_values(by='lift', ascending=False).head(10)
            top10_regras_exibicao = top10_regras[['antecedente','consequente','suporte','confianca','lift']]

            st.subheader(f"Top 10 Regras da base completa (1 antecedente x 1 consequente) — Total de regras geradas: {len(df_regras)}")
            st.dataframe(top10_regras_exibicao)

            # Slider para definir número de períodos
            st.subheader("Definir períodos temporais")
            num_periodos = st.slider("Escolha o número de períodos para particionamento", min_value=2, max_value=10, value=3)

            min_data = df_data_col.min()
            max_data = df_data_col.max()
            delta = (max_data - min_data) / num_periodos
            marcos = [min_data + i*delta for i in range(num_periodos+1)]

            st.write("Marcos temporais definidos automaticamente:")
            st.write([d.strftime("%Y-%m-%d") for d in marcos])

            # Particionar a base por períodos e gerar regras
            st.session_state.bases_periodos = []
            for i in range(num_periodos):
                inicio = marcos[i]
                fim = marcos[i+1]
                periodo_df = df[(df_data_col >= inicio) & (df_data_col <= fim)]
                st.session_state.bases_periodos.append(periodo_df)

                st.write(f"Período {i+1}: {len(periodo_df)} registros ({inicio.date()} a {fim.date()})")

                # Gerar regras para cada período
                df_sem_data_periodo = periodo_df.drop(columns=[selected_date_col])
                df_regras_periodo = analysis.gerar_regras_com_mlxtend(df_sem_data_periodo, sup=min_support, conf=min_confidence)

                if df_regras_periodo is not None and not df_regras_periodo.empty:
                    df_filtrado_periodo = df_regras_periodo[
                        (df_regras_periodo['antecedente'].apply(lambda x: len(x.split(','))==1)) &
                        (df_regras_periodo['consequente'].apply(lambda x: len(x.split(','))==1))
                    ]
                    top10_periodo = df_filtrado_periodo.sort_values(by='lift', ascending=False).head(10)
                    top10_exibicao = top10_periodo[['antecedente','consequente','suporte','confianca','lift']]
                    st.subheader(f"Top 10 Regras do Período {i+1}")
                    st.dataframe(top10_exibicao)
                else:
                    st.warning(f"Nenhuma regra encontrada para o Período {i+1}")



