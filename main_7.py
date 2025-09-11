# main.py
import streamlit as st
import pandas as pd
import analysis  
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ferramenta de Análise Temporal", layout="wide")
st.title("Ferramenta de Análise Temporal de Regras de Associação")

# ---------- Session state ----------
def _init_state():
    defaults = {
        "dados_original": None,
        "dados_processados": None,
        "regras_df": None,
        "min_support": 0.01,
        "min_confidence": 0.5,
        # estado da UI de seleção de regra (atributo -> atributo)
        "regras_selecionadas": [],  # lista de strings "AtribA -> AtribB"
        "campo_antecedente_habilitado": True,
        "campo_consequente_habilitado": True,
        "antecedente_select": "",
        "consequente_select": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ----------------- Abas -----------------
abas = ["Upload e Resumo", "Gerar Regras Gerais", "Particionamento da Base", "Análise Temporal"]
tab = st.tabs(abas)

# ---------- Aba 1: Upload e Resumo ----------
with tab[0]:
    st.header("Upload do CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        st.session_state.dados_original = pd.read_csv(uploaded_file)

        # Resumo
        df = st.session_state.dados_original
        st.success(f"Arquivo carregado com {len(df)} registros e {len(df.columns)} colunas.")
        st.write("Colunas:", df.columns.tolist())

        # Preparar dados (remover datas/números) usando sua função
        df_proc, dados_removidos, atributos_remover = analysis.preparar_dados_para_mineracao_from_df(df)
        st.session_state.dados_processados = df_proc

        # Mostrar o que foi removido
        if atributos_remover:
            st.info(f"Atributos removidos (datas/numéricos): {atributos_remover}")
        else:
            st.info("Nenhum atributo de data/numérico foi removido.")

        st.subheader("Visualização das primeiras linhas (arquivo original)")
        st.dataframe(df.head())

# --- ABA 2: Definição de Regras ---
with tab[1]:
    st.header("2. Definição de Regras")

    # Garantir que os dados foram carregados e processados
    if "dados_processados" in st.session_state and st.session_state.dados_processados is not None:
        
        st.header("Configuração do Apriori")
        min_support = st.number_input(
            "Suporte mínimo (%)", min_value=0.0, max_value=100.0,
            value=st.session_state.min_support * 100
        ) / 100.0
        min_confidence = st.number_input(
            "Confiança mínima (%)", min_value=0.0, max_value=100.0,
            value=st.session_state.min_confidence * 100
        ) / 100.0

        colunas = list(st.session_state.dados_processados.columns)

        # Inicializa estado específico da aba
        if "regras" not in st.session_state:
            st.session_state.regras = []
        if "antecedente_habilitado" not in st.session_state:
            st.session_state.antecedente_habilitado = True
        if "consequente_habilitado" not in st.session_state:
            st.session_state.consequente_habilitado = True
        if "antecedente_select" not in st.session_state:
            st.session_state.antecedente_select = ""
        if "consequente_select" not in st.session_state:
            st.session_state.consequente_select = ""

        # Seleção de antecedente
        antecedente = st.selectbox(
            "Selecione o Antecedente",
            options=[""] + colunas,
            index=0 if "_reset_index" in st.session_state else ([""] + colunas).index(st.session_state.antecedente_select)
            if st.session_state.antecedente_select in ([""] + colunas)
            else 0,
            key="antecedente_select",
            disabled=not st.session_state.antecedente_habilitado
        )

        # Define as opções de consequente
        if "colunas" in locals():
            colunas_lista = colunas
        else:
            colunas_lista = st.session_state.dados_processados.columns.tolist()

        if antecedente:
            opcoes_consequente = [c for c in colunas_lista if c != antecedente]
        else:
            opcoes_consequente = colunas_lista

        # Seleção de consequente
        consequente = st.selectbox(
            "Selecione o Consequente",
            options=[""] + opcoes_consequente,
            index=0 if "_reset_index" in st.session_state else ([""] + opcoes_consequente).index(st.session_state.consequente_select)
            if st.session_state.consequente_select in ([""] + opcoes_consequente)
            else 0,
            key="consequente_select",
            disabled=not st.session_state.consequente_habilitado
        )

        # Limpa flag de reset
        if "_reset_index" in st.session_state:
            del st.session_state._reset_index


        # Botão para adicionar regra
        if st.button("Adicionar Regra"):
            if antecedente and consequente:
                # Adiciona como dicionário
                regra = {"antecedente": antecedente, "consequente": consequente}
                if regra not in st.session_state.regras:
                    st.session_state.regras.append(regra)
                # Desabilita selects até o usuário clicar no "+"
                st.session_state.antecedente_habilitado = False
                st.session_state.consequente_habilitado = False

        # Botão para adicionar nova regra (habilita selects novamente)
        if st.button("➕ Adicionar outra regra"):   
            st.session_state.antecedente_habilitado = True
            st.session_state.consequente_habilitado = True
            # Resetando selects sem sobrescrever a chave
            # Definimos a variável temporária que vai controlar o index do selectbox
            st.session_state._reset_index = True


        # Mostrar regras já adicionadas com opção de remover
        if st.session_state.regras:
            st.subheader("Regras selecionadas:")
            regras_para_remover = []
            for i, regra in enumerate(st.session_state.regras):
                 # regra é um dict com 'antecedente' e 'consequente'
                texto = f"{regra['antecedente']} → {regra['consequente']}"
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.write(texto)
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.regras_selecionadas.pop(i)
                        st.experimental_rerun()  # para atualizar a lista sem duplicar widgets
        
        # Botão para gerar regras filtradas
        if st.button("Gerar regras filtradas"):
            # Gera regras usando sua função do analysis.py
            df_regras = analysis.gerar_regras_com_mlxtend(
                st.session_state.dados_processados,
                sup=st.session_state.min_support,
                conf=st.session_state.min_confidence
            )

            # Salva em sessão para poder usar na tab[2]
            st.session_state.df_regras = df_regras

            df_filtrado = analysis.filtrar_regras_por_atributo(df_regras, st.session_state.regras)
            st.session_state.df_filtrado = df_filtrado
            if df_filtrado.empty:
                st.warning("Nenhuma regra encontrada com os filtros selecionados.")
            else:
                # Mostrar top 10 regras por lift
                top10 = df_filtrado.sort_values(by='lift', ascending=False).head(10)
                st.dataframe(top10[['antecedente', 'consequente', 'suporte', 'confianca', 'lift']])
    else:
        st.warning("⚠️ Por favor, carregue e processe os dados na aba 1 antes de continuar.")


# ---------- Aba 3: PArticionamento da base de dados ----------
with tab[2]:
    st.header("Particionamento da Base")

    # Verifica se o arquivo original foi carregado
    if st.session_state.dados_original is None:
        st.warning("⚠️ Por favor, carregue o arquivo na aba 1 antes de continuar.")
    elif not st.session_state.regras:
        st.warning("⚠️ Nenhuma regra selecionada na aba 2. Selecione ao menos uma regra.")
    else:
        df_original = st.session_state.dados_original.copy()

        # Detectar coluna de data
        col_data = None
        for c in df_original.columns:
            if pd.api.types.is_datetime64_any_dtype(df_original[c]):
                col_data = c
                break
            try:
                converted = pd.to_datetime(df_original[c], errors='coerce')
                if converted.notna().sum() > 0:
                    col_data = c
                    df_original[c] = converted
                    break
            except Exception:
                continue

        if col_data is None:
            st.error("Não foi possível detectar uma coluna de data na base.")
        else:
            st.success(f"Coluna de data detectada: **{col_data}**")

            # Mostrar primeiras e últimas datas
            data_min = df_original[col_data].min()
            data_max = df_original[col_data].max()
            st.write(f"Período da base: **{data_min.date()}** até **{data_max.date()}**")

            # Mostrar regras selecionadas
            if st.session_state.regras:
                st.subheader("Regras selecionadas para análise:")
                for r in st.session_state.regras:
                    st.write(f"- {r}")
            else:
                st.info("Nenhuma regra selecionada na aba 2.")

            st.markdown("---")
            st.subheader("Defina os marcos temporais")

            # Inicializa lista de marcos no session_state
            if "marcos_temporais" not in st.session_state:
                st.session_state.marcos_temporais = []

            # Seleção de data
            novo_marco = st.date_input(
                "Selecione um marco temporal",
                min_value=data_min.date(),
                max_value=data_max.date()
            )

            # Botão para adicionar marco
            if st.button("➕ Adicionar marco"):
                if novo_marco not in st.session_state.marcos_temporais:
                    st.session_state.marcos_temporais.append(novo_marco)
                    st.success(f"Marco {novo_marco} adicionado!")
                else:
                    st.warning("Este marco já foi adicionado.")

            # Mostrar lista de marcos já adicionados com opção de remover
            if st.session_state.marcos_temporais:
                st.write("Marcos temporais definidos:")
                for i, m in enumerate(sorted(st.session_state.marcos_temporais)):
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.write(f"- {m}")
                    with col2:
                        if st.button("❌", key=f"remove_marco_{i}"):
                            st.session_state.marcos_temporais.pop(i)
        
            st.subheader("Particionar dados por marcos temporais")

            if st.button("📊 Particionar dados"):
                if "marcos_temporais" not in st.session_state or not st.session_state.marcos_temporais:
                    st.warning("⚠️ Nenhum marco temporal definido. Por favor, adicione ao menos um marco.")
                else:
                    df_original = st.session_state.dados_original.copy()

                    # Garante que a coluna de datas está em datetime
                    col_data = None
                    for c in df_original.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_original[c]):
                            col_data = c
                            break
                        try:
                            converted = pd.to_datetime(df_original[c], errors='coerce')
                            if converted.notna().sum() > 0:
                                col_data = c
                                df_original[c] = converted
                                break
                        except Exception:
                            continue

                    if col_data is None:
                        st.error("Não foi possível detectar uma coluna de data na base.")
                    else:
                        # Converte os marcos para datetime também
                        marcos = [pd.to_datetime(m) for m in sorted(st.session_state.marcos_temporais)]
                        data_min = df_original[col_data].min()
                        data_max = df_original[col_data].max()

                        # Limites de cada partição
                        limites = [data_min] + marcos + [data_max]

                        particoes = []
                        for i in range(len(limites)-1):
                            inicio = limites[i]
                            fim = limites[i+1]
                            # A segunda partição em diante começa 1 dia depois do fim anterior
                            if i > 0:
                                inicio += pd.Timedelta(days=1)
                            part = df_original[
                                (df_original[col_data] >= inicio) &
                                (df_original[col_data] <= fim)
                            ].copy()
                            particoes.append({"inicio": inicio, "fim": fim, "dados": part})

                        # Salva no session_state
                        st.session_state.particoes_temporais = particoes

                        # Mostrar resumo
                        st.success(f"Particionamento concluído! Total de partições: {len(particoes)}")
                        for i, p in enumerate(particoes):
                            st.write(f"Partição {i+1}: {p['inicio'].date()} até {p['fim'].date()} — {len(p['dados'])} registros")

            st.subheader("Análise Temporal das Regras")

            if "particoes_temporais" not in st.session_state or not st.session_state.particoes_temporais:
                st.warning("⚠️ Por favor, particione os dados antes de gerar a análise temporal.")
            elif not st.session_state.regras:
                st.warning("⚠️ Nenhuma regra selecionada para análise temporal.")
            else:
                if st.button("📊 Gerar Análise Temporal"):
                    resultados = []
                    col_data = None
                    # Detectar a coluna de data dinamicamente na primeira partição
                    for c in st.session_state.particoes_temporais[0]["dados"].columns:
                        if pd.api.types.is_datetime64_any_dtype(st.session_state.particoes_temporais[0]["dados"][c]):
                            col_data = c
                            break

                    for i, part in enumerate(st.session_state.particoes_temporais):
                        df_part = part["dados"].copy()
                        if df_part.empty:
                            st.warning(f"Partição {i+1} está vazia, pulando.")
                            continue

                        # Remove coluna de data antes de gerar regras
                        if col_data and col_data in df_part.columns:
                            df_part = df_part.drop(columns=[col_data])

                        # Gera regras na partição
                        df_regras_part = analysis.gerar_regras_com_mlxtend(
                            df_part, 
                            st.session_state.min_support, 
                            st.session_state.min_confidence
                        )

                        part_result = {"particao": i+1, "inicio": part["inicio"], "fim": part["fim"], "regras": []}

                        for r in st.session_state.regras:
                            antecedente = r["antecedente"]
                            consequente = r["consequente"]

                            df_filtrada = analysis.filtrar_regras_por_item(
                                df_regras_part,
                                antecedentes=[antecedente],
                                consequentes=[consequente]
                            )

                            def agreg_metrics(df):
                                if df.empty:
                                    return {"suporte": 0, "confianca": 0, "lift": 0}
                                else:
                                    return {
                                        "suporte": df["suporte"].mean(),
                                        "confianca": df["confianca"].mean(),
                                        "lift": df["lift"].mean()
                                    }

                            metrics_part = agreg_metrics(df_filtrada)

                            part_result["regras"].append({
                                "antecedente": antecedente,
                                "consequente": consequente,
                                "suporte": metrics_part["suporte"],
                                "confianca": metrics_part["confianca"],
                                "lift": metrics_part["lift"]
                            })

                        resultados.append(part_result)

                    # Mostrar resultados numéricos
                    for part in resultados:
                        st.subheader(f"Partição {part['particao']} — {part['inicio'].date()} até {part['fim'].date()}")
                        for r in part["regras"]:
                            st.write(f"**{r['antecedente']} -> {r['consequente']}**")
                            st.write(f"- Suporte: {r['suporte']:.3f}")
                            st.write(f"- Confiança: {r['confianca']:.3f}")
                            st.write(f"- Lift: {r['lift']:.3f}")
                    
                    # === Gerar Gráficos Comparativos ===
                    st.subheader("📈 Comparação Visual das Medidas")
                    
                    if "df_regras" in st.session_state and st.session_state.regras:
                        for regra in st.session_state.regras:
                            antecedente = regra["antecedente"]
                            consequente = regra["consequente"]

                            # Filtra a regra na base completa
                            df_geral = st.session_state.df_regras
                            df_filtrada_geral = analysis.filtrar_regras_por_item(
                                df_geral,
                                antecedentes=[antecedente],
                                consequentes=[consequente]
                            )

                            if df_filtrada_geral.empty:
                                suporte_geral = confianca_geral = lift_geral = 0
                            else:
                                suporte_geral = df_filtrada_geral["suporte"].mean()
                                confianca_geral = df_filtrada_geral["confianca"].mean()
                                lift_geral = df_filtrada_geral["lift"].mean()

                            # Extrai valores por partição
                            suporte_part = [p["regras"][0]["suporte"] for p in resultados]
                            confianca_part = [p["regras"][0]["confianca"] for p in resultados]
                            lift_part = [p["regras"][0]["lift"] for p in resultados]
                            labels = [f"P{i+1}" for i in range(len(resultados))]

                            # Gráfico de Suporte
                            fig, ax = plt.subplots(figsize=(2,1))
                            ax.bar(labels, suporte_part, color="skyblue")
                            ax.axhline(y=suporte_geral, color="red", linestyle="--", label="Base Geral")
                            ax.set_title(f"Suporte — Regra {antecedente} -> {consequente}")
                            ax.set_ylabel("Suporte")
                            ax.legend()
                            st.pyplot(fig)

                            # Gráfico de Confiança
                            fig, ax = plt.subplots(figsize=(2,1))
                            ax.bar(labels, confianca_part, color="lightgreen")
                            ax.axhline(y=confianca_geral, color="red", linestyle="--", label="Base Geral")
                            ax.set_title(f"Confiança — Regra {antecedente} -> {consequente}")
                            ax.set_ylabel("Confiança")
                            ax.legend()
                            st.pyplot(fig)

                            # Gráfico de Lift
                            fig, ax = plt.subplots(figsize=(2,1))
                            ax.bar(labels, lift_part, color="orange")
                            ax.axhline(y=lift_geral, color="red", linestyle="--", label="Base Geral")
                            ax.set_title(f"Lift — Regra {antecedente} -> {consequente}")
                            ax.set_ylabel("Lift")
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        st.warning("Você precisa gerar as regras na aba anterior e selecionar ao menos uma regra para análise.")
# ---------- Aba 4: Análise temporal ----------
with tab[3]:
    st.header("Análise Temporal")

    if "particoes_temporais" not in st.session_state:
        st.session_state.particoes_temporais = []

    # Verifica se o arquivo original foi carregado
    if st.session_state.dados_original is None:
        st.warning("⚠️ Por favor, carregue o arquivo na aba 1 antes de continuar.")
    elif not st.session_state.regras:
        st.warning("⚠️ Nenhuma regra selecionada na aba 2. Selecione ao menos uma regra.")
    elif not st.session_state.particoes_temporais:
        st.warning("⚠️ O Particionamento não foi realizado. Particione a base na aba 3.")
    else:
        st.subheader("Análise Temporal das Regras")