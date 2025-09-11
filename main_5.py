# main.py
import streamlit as st
import pandas as pd
import analysis  # seu analysis.py

st.set_page_config(page_title="Ferramenta de Análise Temporal", layout="wide")

st.title("Ferramenta de Análise Temporal de Regras de Associação")

# Inicializa estado
if 'dados' not in st.session_state:
    st.session_state.dados = None
if 'regras' not in st.session_state:
    st.session_state.regras = None
if 'min_support' not in st.session_state:
    st.session_state.min_support = 0.01
if 'min_confidence' not in st.session_state:
    st.session_state.min_confidence = 0.5

# ----------------- Abas -----------------
abas = ["Upload e Resumo", "Gerar Regras", "Top 10 Regras"]
tab = st.tabs(abas)

# ---------- Aba 1: Upload e Resumo ----------
with tab[0]:
    st.header("Upload do CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file is not None:
        st.session_state.dados = pd.read_csv(uploaded_file)

        # Processa os dados com a função do analysis.py
        df_discretizado, dados_removidos, atributos_remover = analysis.preparar_dados_para_mineracao_from_df(st.session_state.dados)

        # Guarda no state para uso posterior
        st.session_state.dados_processados = df_discretizado
        st.session_state.dados_removidos = dados_removidos
        st.session_state.atributos_remover = atributos_remover

        # Mensagem de resumo
        st.success(f"Arquivo carregado com {len(st.session_state.dados)} registros e {len(st.session_state.dados.columns)} colunas.")

        # Mostra atributos removidos
        if atributos_remover:
            st.warning(f"Atributos removidos do processamento: {atributos_remover}")
        else:
            st.info("Nenhum atributo foi removido do processamento.")

        # Mostra colunas do dataset original
        st.write("Colunas originais:", st.session_state.dados.columns.tolist())

        # Visualização
        st.subheader("Visualização das primeiras linhas (dataset original)")
        st.dataframe(st.session_state.dados.head())


# ---------- Aba 2: Gerar Regras ----------
with tab[1]:
    if st.session_state.dados is None:
        st.warning("Por favor, carregue o arquivo na aba 'Upload e Resumo' primeiro.")
    else:
        st.header("Configuração do Apriori")
        min_support = st.number_input(
            "Suporte mínimo (%)", min_value=0.0, max_value=100.0,
            value=st.session_state.min_support * 100
        ) / 100.0
        min_confidence = st.number_input(
            "Confiança mínima (%)", min_value=0.0, max_value=100.0,
            value=st.session_state.min_confidence * 100
        ) / 100.0


        # Cria lista de opções a partir dos dados brutos
        itens_disponiveis = analysis.listar_itens_possiveis(st.session_state.dados_processados)

        antecedentes_selecionados = st.multiselect(
            "Selecione os antecedentes (atributo=valor)",
            options=itens_disponiveis
        )

        consequentes_selecionados = st.multiselect(
            "Selecione os consequentes (atributo=valor)",
            options=itens_disponiveis
        )


        # Botão para gerar regras filtradas
        if st.button("Gerar regras filtradas"):
            # Gera regras usando sua função do analysis.py
            df_regras = analysis.gerar_regras_com_mlxtend(
                st.session_state.dados_processados,
                sup=st.session_state.min_support,
                conf=st.session_state.min_confidence
            )

            # Filtra regras pelo antecedente e consequente selecionados
            if antecedentes_selecionados or consequentes_selecionados:
                df_regras = analysis.filtrar_regras_por_item(
                    df_regras,
                    antecedentes=antecedentes_selecionados,
                    consequentes=consequentes_selecionados
                )

            if df_regras.empty:
                st.warning("Nenhuma regra encontrada com os filtros selecionados.")
            else:
                # Mostrar top 10 regras por lift
                top10 = df_regras.sort_values(by='lift', ascending=False).head(10)
                st.dataframe(top10[['antecedente', 'consequente', 'suporte', 'confianca', 'lift']])

# ---------- Aba 3: Top 10 Regras ----------
with tab[2]:
    if st.session_state.regras is None:
        st.warning("Por favor, gere as regras na aba 'Gerar Regras' primeiro.")
    else:
        st.header("Top 10 Regras (1 antecedente x 1 consequente)")
        # Filtrar regras 1x1
        df_filtrado = st.session_state.regras[
            (st.session_state.regras['antecedente'].apply(lambda x: len(x.split(',')) == 1)) &
            (st.session_state.regras['consequente'].apply(lambda x: len(x.split(',')) == 1))
        ]
        top10_regras = df_filtrado.sort_values(by='lift', ascending=False).head(10)
        top10_exibicao = top10_regras[['antecedente', 'consequente', 'suporte', 'confianca', 'lift']]

        st.write(f"Total de regras geradas: {len(st.session_state.regras)}")
        st.dataframe(top10_exibicao)

        st.download_button(
            label="Baixar todas as regras em Excel",
            data=analysis.exportar_regras_para_excel_bytes(st.session_state.regras),
            file_name="regras.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )














