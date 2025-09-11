# main.py
import streamlit as st
import pandas as pd
import analysis  # seu analysis.py

st.set_page_config(page_title="Análise Temporal", layout="wide")
st.title("Ferramenta de Análise Temporal de Pull Requests")

# --- Upload do CSV ---
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(f"Visualização das primeiras linhas (total de {len(df)} registros)")
    st.dataframe(df.head())

    # Detectar colunas de data
    datetime_cols = []
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            ratio = converted.notna().mean()
            if ratio > 0.7:
                datetime_cols.append(col)
        except Exception:
            continue

    if not datetime_cols:
        st.error("Nenhuma coluna de data detectada.")
    else:
        st.subheader("Colunas de data detectadas")
        selected_cols = st.multiselect(
            "Selecione as colunas de data para análise temporal",
            datetime_cols,
            default=datetime_cols
        )
        df_data = df[selected_cols] if selected_cols else None
        df_sem_data = df.drop(columns=selected_cols)

        st.subheader("Colunas usadas para mineração")
        st.write(df_sem_data.columns.tolist())

        # --- Parâmetros de mineração ---
        st.subheader("Configuração do Apriori")
        min_support = st.number_input("Suporte mínimo (%)", min_value=0.0, max_value=100.0, value=1.0) / 100.0
        min_confidence = st.number_input("Confiança mínima (%)", min_value=0.0, max_value=100.0, value=50.0) / 100.0

        if st.button("Gerar regras"):
            df_regras = analysis.gerar_regras_com_mlxtend(df_sem_data, sup=min_support, conf=min_confidence)
            st.session_state['df_regras'] = df_regras

        # --- Exibir regras geradas ---
        if 'df_regras' in st.session_state and not st.session_state['df_regras'].empty:
            df_regras = st.session_state['df_regras']

            # Filtrar 1 antecedente x 1 consequente
            df_filtrado = df_regras[
                (df_regras['antecedente'].apply(lambda x: len(x.split(',')) == 1)) &
                (df_regras['consequente'].apply(lambda x: len(x.split(',')) == 1))
            ]
            top10_regras = df_filtrado.sort_values(by='lift', ascending=False).head(10)
            st.subheader(f"Top 10 Regras (1 antecedente x 1 consequente) - Total de regras: {len(df_regras)}")
            st.dataframe(top10_regras[['antecedente', 'consequente', 'suporte', 'confianca', 'lift']])

            # Botão para baixar todas as regras
            excel_bytes = analysis.exportar_regras_para_excel_bytes(df_regras)
            st.download_button(
                label="Baixar todas as regras em Excel",
                data=excel_bytes,
                file_name="regras.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # --- Particionamento temporal ---
            if df_data is not None:
                st.subheader("Particionamento temporal")

                min_date = pd.to_datetime(df_data.min().min(), errors='coerce')
                max_date = pd.to_datetime(df_data.max().max(), errors='coerce')


                # Seletor de número de partições
                num_particoes = st.slider("Escolha o número de partições automáticas", 2, 12, 3)

                # Criar marcos automáticos
                marcos = pd.date_range(start=min_date, end=max_date, periods=num_particoes + 1)[1:-1]
                st.write(f"Marcos temporais automáticos ({num_particoes} partições):", marcos.date)

                # Seleção de datas manual (calendar) simplificada
                st.subheader("Escolha manualmente marcos temporais (opcional)")
                datas_manual = st.date_input(
                    "Selecione marcos adicionais",
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    value=[]
                )
                if datas_manual:
                    st.write("Marcos manuais selecionados:", datas_manual)






