# main.py
import streamlit as st
import pandas as pd
import analysis  
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
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
        
        # Preparar dados (remover datas/números) usando sua função
        df_proc, dados_removidos, atributos_remover = analysis.preparar_dados_para_mineracao_from_df(df)
        st.session_state.dados_processados = df_proc

        # Mostrar o que foi removido
        if atributos_remover:
            st.info(f"Atributos removidos (datas/numéricos): {atributos_remover}")
        else:
            st.info("Nenhum atributo de data/numérico foi removido.")
        
        #Resumo dos atributos
        st.subheader("Resumo das Colunas (valores únicos e frequência)")
        
        cols = st.columns(3)  # cria 3 colunas
        col_index = 0  # controla em qual coluna inserir
        
        for col in df_proc.columns:
                    
            # Pega os 10 valores mais comuns
            value_counts = df_proc[col].value_counts(dropna=False).head(10)

            # Cria figura
            fig, ax = plt.subplots(figsize=(2, 1.5))  # 👈 gráfico pequeno
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, color="skyblue")

            # Texto em cima das barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, str(height),
                        ha="center", va="bottom", fontsize=5)

            # ajusta limite do eixo Y para dar espaço
            max_val = value_counts.max()
            ax.set_ylim(0, max_val * 1.15)  # 15% acima da barra mais alta
            
            # Ajustes visuais
            ax.tick_params(axis="x", labelsize=5, rotation=45)
            ax.tick_params(axis="y", labelsize=5)
            plt.tight_layout(pad=0.3)
            # coloca o gráfico na coluna atual
            with cols[col_index]:
                st.markdown(
                    f"<p style='text-align: center; font-weight: bold;'>{col}</p>",
                    unsafe_allow_html=True
                )
                st.pyplot(fig, use_container_width=False)
            
            # avança para a próxima coluna
            col_index += 1

            # se já preencheu 3 colunas, cria nova linha
            if col_index == 3:
                st.markdown("<div style='height:1px; background-color:#e0e0e0; margin:5px 0;'></div>",
                    unsafe_allow_html=True)
                cols = st.columns(3)
                col_index = 0

            total_valores = df_proc[col].nunique(dropna=False)
            if total_valores > 10:
                st.info(f"⚠️ Atributo `{col}` possui {total_valores} valores únicos. Exibindo apenas os 10 mais frequentes.")

        st.subheader("Visualização das primeiras linhas (arquivo original)")
        st.dataframe(df_proc.head())

# --- ABA 2: Definição de Regras ---
with tab[1]:
    st.header("Configuração do algoritmo e Definição das Regras")
    
    # Garantir que os dados foram carregados e processados
    if "dados_processados" in st.session_state and st.session_state.dados_processados is not None:

        st.subheader("Configuração do Apriori")
        
        # Cria 3 colunas: esquerda, central e direita
        col_esq, col_central, col_dir = st.columns([2, 1, 1])

        with col_esq:
            # Inputs lado a lado dentro da coluna central
            col_s, col_c = st.columns([1,1])
        with col_s:
            min_support = st.number_input(
                "Suporte mínimo (%)", min_value=0.0, max_value=100.0,
                value=st.session_state.min_support * 100
            ) / 100.0
        with col_c:
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

        st.subheader("Definição das Regras para Análise")
        
        # Cria 3 colunas: esquerda, central e direita
        col_e, col_cent, col_d = st.columns([2, 1, 1])

        with col_e:
            # Inputs lado a lado dentro da coluna central
            col_1, col_2 = st.columns([1,1])
        with col_1:
            
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

        with col_2:
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
        if st.button("Adicionar meta regra"):
            if antecedente and consequente:
                # Adiciona como dicionário
                regra = {"antecedente": antecedente, "consequente": consequente}
                if regra not in st.session_state.regras:
                    st.session_state.regras.append(regra)
                # Desabilita selects até o usuário clicar no "+"
                st.session_state.antecedente_habilitado = False
                st.session_state.consequente_habilitado = False

        # Botão para adicionar nova regra (habilita selects novamente)
        if st.button("➕ Compor nova meta regra"):   
            st.session_state.antecedente_habilitado = True
            st.session_state.consequente_habilitado = True
            # Resetando selects sem sobrescrever a chave
            # Definimos a variável temporária que vai controlar o index do selectbox
            st.session_state._reset_index = True

        # Criar lista temporária para reconstruir sem o item removido
        novas_regras = []

        # Mostrar regras já adicionadas com opção de remover
        if st.session_state.regras:
            st.subheader("Meta regras selecionadas:")
            
            for i, regra in enumerate(st.session_state.regras):
                texto = f"{regra['antecedente']} → {regra['consequente']}"
                col1, col2 = st.columns([9, 1])
                remover = False
                with col1:
                    st.write(texto)
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        remover = True
                
                # Só mantém a regra se não clicou no X
                if not remover:
                    novas_regras.append(regra)

        # Atualiza a lista de regras no session_state
        st.session_state.regras = novas_regras
        
        # Botão para gerar regras filtradas
        if st.button("Gerar regras"):
            df_regras = analysis.gerar_regras_com_mlxtend(
                st.session_state.dados_processados,
                sup=st.session_state.min_support,
                conf=st.session_state.min_confidence
            )
            st.session_state.df_regras = df_regras

            df_filtrado = analysis.filtrar_regras_por_atributo(df_regras, st.session_state.regras)
            st.session_state.df_filtrado = df_filtrado
            
            df_oht = analysis.preparar_para_apriori(st.session_state.dados_processados)
            frequent_itemsets = apriori(df_oht, min_support=st.session_state.min_support, use_colnames=True)
            st.write(frequent_itemsets.sort_values('support', ascending=False))
            
            st.write(float(st.session_state.min_confidence))
            # Mostrar as regras na tela
            # --- Análise Geral por regra escolhida (sem médias; valores exatos) ---
            if df_filtrado.empty:
                st.warning("Nenhuma regra encontrada com os filtros selecionados.")
            else:
                st.subheader("Análise Geral das Regras")
                
                #st.write("Regras selecionadas pelo usuário:", st.session_state.regras)

                st.subheader("Todas as regras geradas")
                st.dataframe(st.session_state.df_regras)
                
                df_teste = st.session_state.df_regras[
                    st.session_state.df_regras["consequente"].str.contains("merged=False", na=False)
                ]

                st.subheader("Regras teste")
                st.dataframe(df_teste)
                
                df_test = st.session_state.dados_processados
                contagem = df_test.groupby(['firstpull', 'merged']).size().reset_index(name='qtd')
                st.write("Contagem de PRs por firstpull e merged:")
                st.dataframe(contagem)

                # Normaliza nomes, se vierem com maiúsculas
                df_plot = df_filtrado.rename(columns={
                    "Suporte": "suporte",
                    "Confianca": "confianca",
                    "Lift": "lift",
                    "Antecedente": "antecedente",
                    "Consequente": "consequente",
                })

                # Para cada regra definida pelo usuário, gere seus gráficos isoladamente
                for regra_user in st.session_state.regras:
                    atributo_antecedente = regra_user["antecedente"]
                    atributo_consequente = regra_user["consequente"]

                    # Seleciona apenas regras simples no antecedente (sem vírgula)
                    mask_ant = st.session_state.df_regras["antecedente"].str.match(f"^{atributo_antecedente}=.+$") & \
                            (~st.session_state.df_regras["antecedente"].str.contains(","))

                    # Seleciona apenas regras simples no consequente
                    mask_cons = st.session_state.df_regras["consequente"].str.match(f"^{atributo_consequente}=.+$") & \
                            (~st.session_state.df_regras["consequente"].str.contains(","))

                    base_regra = st.session_state.df_regras[mask_ant & mask_cons].copy()
                    st.write("Regras filtradas:", base_regra)             
                    if base_regra.empty:
                        st.warning(f"Nenhuma regra encontrada para {atributo_antecedente} → {atributo_consequente}")
                        continue

                    st.subheader(f"Regra: {atributo_antecedente} → {atributo_consequente}")
                    
                    # 3 gráficos lado a lado: Suporte, Confiança, Lift
                    cols = st.columns(3)
                    for i, medida in enumerate(["suporte", "confianca", "lift"]):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(2.2, 1.6))
                            bars = ax.bar(base_regra["antecedente"].astype(str) + " → " + base_regra["consequente"].astype(str),
                                        base_regra[medida])

                            # estética compacta
                            ymax = float(base_regra[medida].max())
                            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
                            ax.tick_params(axis="x", labelsize=6, rotation=45)
                            ax.tick_params(axis="y", labelsize=6)
                            ax.set_title(medida.capitalize(), fontsize=8, pad=2)

                            # valores em cima das barras
                            for b in bars:
                                h = b.get_height()
                                ax.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}",
                                        ha="center", va="bottom", fontsize=6)

                            plt.tight_layout(pad=0.3)
                            st.pyplot(fig, use_container_width=False)

                    # separador fino entre linhas (cada valor do consequente)
                    st.markdown(
                        "<div style='height:1px; background:#e6e6e6; margin:6px 0;'></div>",
                        unsafe_allow_html=True
                    )          
    

# ---------- Aba 3: Particionamento da base de dados ----------
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

        if "particoes_temporais" not in st.session_state or not st.session_state.particoes_temporais:
            st.warning("⚠️ Por favor, particione os dados antes de gerar a análise temporal.")
        elif not st.session_state.regras:
            st.warning("⚠️ Nenhuma regra selecionada para análise temporal.")
        else:
            if st.button("📊 Gerar Análise Temporal"):
                resultados = []
                col_data = None

                # Detecta a coluna de data
                for c in st.session_state.particoes_temporais[0]["dados"].columns:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.particoes_temporais[0]["dados"][c]):
                        col_data = c
                        break

                user_rule = st.session_state.regras[0]
                ant_attr = user_rule["antecedente"]
                cons_attr = user_rule["consequente"]

                for i, part in enumerate(st.session_state.particoes_temporais):
                    df_part = part["dados"].copy()
                    if df_part.empty:
                        st.warning(f"Partição {i+1} está vazia, pulando.")
                        continue

                    if col_data and col_data in df_part.columns:
                        df_part = df_part.drop(columns=[col_data])

                    df_regras_part = analysis.gerar_regras_com_mlxtend(
                        df_part,
                        st.session_state.min_support,
                        st.session_state.min_confidence
                    )

                    # Normaliza
                    df_regras_part["antecedente"] = df_regras_part["antecedente"].apply(lambda s: analysis.normalizar_regra(s, df_part=df_part))
                    df_regras_part["consequente"] = df_regras_part["consequente"].apply(lambda s: analysis.normalizar_regra(s, df_part=df_part))

                    st.subheader(f"Partição {i+1}: regras geradas")
                    st.dataframe(df_regras_part)  # Mostra o que foi gerado

                    # Filtra regra simples do usuário
                    df_filtrado = df_regras_part[
                        df_regras_part["antecedente"].str.contains(f"{ant_attr}") &
                        df_regras_part["consequente"].str.contains(f"{cons_attr}")
                    ]
                    st.subheader(f"Partição {i+1}: regras filtradas para {ant_attr} -> {cons_attr}")
                    st.dataframe(df_filtrado)  # Mostra o que sobrou após o filtro

                    part_result = {"particao": f"Part {i+1}", "regras": df_filtrado.to_dict('records')}
                    resultados.append(part_result)

                st.write("Resultados finais:", resultados)