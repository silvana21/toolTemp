# main.py
import streamlit as st
import pandas as pd
import analysis  
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import re
from dateutil.relativedelta import relativedelta
import time
import plotly.express as px

st.set_page_config(page_title="Análise Temporal de Regras de Associação", layout="wide")

#st.title("Análise Temporal de Regras de Associação")

# ---------- Session state ----------
def _init_state():
    defaults = {
        "dados_original": None,
        "dados_processados": None,
        "regras_df": None,
        "min_support": 0.01,
        "min_confidence": 0.01,
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

# --- Estado persistente de ordenação (usado entre abas) ---
if "ordem_valores" not in st.session_state:
    st.session_state["ordem_valores"] = {}

def _sanitize_input(key, min_value=None, max_value=None, decimals=1):
    """Callback: limpa st.session_state[key] mantendo só números e 1 ponto decimal."""
    s = st.session_state.get(key, "")
    s = s.strip().replace(",", ".")
    # remove tudo que não seja dígito ou ponto
    s = re.sub(r"[^0-9.]", "", s)
    # mantém apenas o primeiro ponto
    if s.count(".") > 1:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])
    # caso vazio ou apenas ponto, deixa vazio (campo ficará vazio)
    if s in ("", "."):
        st.session_state[key] = ""
        return
    try:
        num = float(s)
    except ValueError:
        st.session_state[key] = ""
        return
    # limites
    if (min_value is not None) and (num < min_value):
        num = min_value
    if (max_value is not None) and (num > max_value):
        num = max_value
    # formata com casas decimais desejadas
    fmt = f"{{:.{decimals}f}}".format(num)
    st.session_state[key] = fmt

def numeric_text_input(label, key, value=0.0, min_value=None, max_value=None, decimals=1, width=90):
    """
    Text input que aceita apenas números (limpa via on_change).
    Retorna float (valor default se campo vazio ou inválido).
    - key: chave única no session_state
    - value: valor inicial (float)
    - min_value/max_value: limites opcionais
    - decimals: casas decimais exibidas
    """
    # inicializa state se necessário
    if key not in st.session_state:
        st.session_state[key] = f"{value:.{decimals}f}"

    # cria o text_input que dispara _sanitize_input ao alterar (quando o widget perde foco / Enter)
    st.text_input(
        label,
        #value=st.session_state[key],
        key=key,
        on_change=_sanitize_input,
        args=(key, min_value, max_value, decimals)
    )

    # retorna float limpo (ou valor padrão)
    s = st.session_state.get(key, "")
    try:
        return float(s) if s != "" else float(value)
    except Exception:
        return float(value)

# ----------------- Abas -----------------
# --- MENU LATERAL ---
with st.sidebar:
    st.markdown("### 🧭 Navegação")

    tab = option_menu(
        None,
        ["Carregar CSV", "Regras Gerais", "Partições", "Análise Temporal"],
        icons=["file-earmark-arrow-up", "diagram-3", "calendar3", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important",
            "background-color": "transparent"},
            "icon": {"color": "#4a4a4a", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "2px 0",
                "--hover-color": "#f0f2f6",
            },
            "nav-link-selected": {
                "background-color": "#0366d6",
                "color": "white",
                "font-weight": "bold",
            },
        },
    )

# ---------- Aba 1: Upload e Resumo ----------
if tab == "Carregar CSV":
    st.subheader("Carregar CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", key="uploader_csv")

    # 1) Se houve upload e é um arquivo novo → processa e salva tudo no session_state
    if uploaded_file is not None:
        if ("arquivo_carregado" not in st.session_state) or (uploaded_file.name != st.session_state["arquivo_carregado"]):
            st.session_state["arquivo_carregado"] = uploaded_file.name
            st.session_state.dados_original = pd.read_csv(uploaded_file)

            df = st.session_state.dados_original
            
            # Prepara dados para mineração e salva no estado
            df_proc, dados_removidos, atributos_remover = analysis.preparar_dados_para_mineracao_from_df(df)
            st.session_state.dados_processados = df_proc
            st.session_state["atributos_removidos"] = atributos_remover

            # Pré-calcula resumos e salva
            st.session_state["resumo_colunas"] = {
                col: df_proc[col].value_counts(dropna=False).head(10)
                for col in df_proc.columns
            }
        else:
            st.info(f"Mantendo arquivo atual: **{st.session_state['arquivo_carregado']}**")

    # 2) Renderiza SEMPRE que houver dados no estado (mesmo sem upload nesta reexecução)
    if st.session_state.get("dados_processados") is not None:
        df_proc = st.session_state.dados_processados
        atributos_remover = st.session_state.get("atributos_removidos", [])

        # garante que temos os resumos (se não houver por algum motivo, gera agora)
        if "resumo_colunas" not in st.session_state or not st.session_state["resumo_colunas"]:
            st.session_state["resumo_colunas"] = {
                col: df_proc[col].value_counts(dropna=False).head(10)
                for col in df_proc.columns
            }

        if "arquivo_carregado" in st.session_state:
            st.success(
                f"Arquivo atual: **{st.session_state['arquivo_carregado']}** "
                f"({len(st.session_state.dados_original)} registros, {len(st.session_state.dados_original.columns)} colunas)"
            )

        if atributos_remover:
            st.info(f"Atributos removidos (datas/numéricos): {atributos_remover}")
        else:
            st.info("Nenhum atributo de data/numérico foi removido.")

        st.subheader("Resumo dos Atributos (valores e frequência)")

        # --- guarda ordens personalizadas por atributo ---
        if "ordem_valores" not in st.session_state:
            st.session_state["ordem_valores"] = {}

        # ====================================================
        # 1️⃣ PRIMEIRO LOOP → apenas interfaces de ordenação
        # ====================================================
        st.markdown("### Defina a ordem de exibição dos valores")
        try:
            from streamlit_sortables import sort_items  # type: ignore
            usa_dragdrop = True
        except Exception:
            usa_dragdrop = False

        for col, value_counts in st.session_state["resumo_colunas"].items():
            lista_valores = [str(v) for v in value_counts.index.tolist()]

            if usa_dragdrop:
                st.caption(f"Defina a ordem de exibição para **{col}** (arraste para reorganizar)")

                # 🔹 Recupera a ordem salva anteriormente, se existir
                ordem_salva = st.session_state["ordem_valores"].get(col, lista_valores)

                # 🔹 Garante que os itens que sumiram (novos valores, etc.) ainda apareçam
                for item in lista_valores:
                    if item not in ordem_salva:
                        ordem_salva.append(item)

                # 🔹 Mostra o componente já com a ordem salva
                ordem_escolhida = sort_items(
                    items=ordem_salva,
                    direction="vertical",
                    key=f"sort_{col}"
                )

                # 🔹 Se o usuário não interagiu (ordem_escolhida vazia), mantém a salva
                if not ordem_escolhida:
                    ordem_escolhida = ordem_salva
            else:
                st.caption(f"Defina a ordem de exibição para **{col}** (selecione na ordem desejada)")
                ordem_escolhida = st.multiselect(
                    label=f"Ordem de {col}",
                    options=lista_valores,
                    default=st.session_state["ordem_valores"].get(col, lista_valores),
                    key=f"ms_{col}"
                )
                if not ordem_escolhida:
                    ordem_escolhida = lista_valores

            # Salva de forma persistente (mantém entre abas)
            if "ordem_valores" not in st.session_state:
                st.session_state["ordem_valores"] = {}

            st.session_state["ordem_valores"][col] = ordem_escolhida.copy()

        st.markdown("---")  # separador visual antes dos gráficos

        # ====================================================
        # 2️⃣ SEGUNDO LOOP → gera os gráficos com base nas ordens
        # ====================================================
        st.markdown("### Visualização dos atributos")

        cols = st.columns(3)
        col_index = 0

        for col, value_counts in st.session_state["resumo_colunas"].items():
            ordem_escolhida = st.session_state["ordem_valores"].get(
                col, value_counts.index.tolist()
            )
            # aplica ordem escolhida
            value_counts = value_counts.reindex(ordem_escolhida)

            fig, ax = plt.subplots(figsize=(2, 1.5))
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, color="skyblue")

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, str(h),
                        ha="center", va="bottom", fontsize=5)

            max_val = value_counts.max() if len(value_counts) else 0
            ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)
            ax.tick_params(axis="x", labelsize=5, rotation=45)
            ax.tick_params(axis="y", labelsize=5)
            plt.tight_layout(pad=0.3)

            with cols[col_index]:
                st.markdown(
                    f"<p style='text-align: center; font-weight: bold;'>{col}</p>",
                    unsafe_allow_html=True
                )
                st.pyplot(fig, use_container_width=False)

            col_index += 1
            if col_index == 3:
                st.markdown(
                    "<div style='height:1px; background-color:#e0e0e0; margin:5px 0;'></div>",
                    unsafe_allow_html=True
                )
                cols = st.columns(3)
                col_index = 0

            total_valores = df_proc[col].nunique(dropna=False)
            if total_valores > 10:
                st.info(f"Atributo `{col}` possui {total_valores} valores. Exibindo apenas os 10 mais frequentes.")
    # 3) Caso não tenha nada ainda
    else:
        st.warning("Nenhum arquivo CSV foi carregado ainda.")

# --- ABA 2: Definição de Regras ---
elif tab == "Regras Gerais":
    
    st.subheader("Configuração do algoritmo")
    
    # Garantir que os dados foram carregados e processados
    if "dados_processados" in st.session_state and st.session_state.dados_processados is not None:

        # Cria 3 colunas: esquerda, central e direita
        col_esq, col_central, col_dir = st.columns([2, 1, 1])

        with col_esq:
            # Inputs lado a lado dentro da coluna central
            col_s, col_c = st.columns([1,1])
        with col_s:
            min_support_pct = numeric_text_input(
                "Suporte mínimo (%)",
                key="min_support_input",
                value=st.session_state.min_support * 100,
                min_value=0.0,
                max_value=100.0
            )
            min_support = min_support_pct / 100.0
        with col_c:
            min_confidence_pct = numeric_text_input(
                "Confiança mínima (%)",
                key="min_confidence_input",
                value=st.session_state.min_confidence * 100,
                min_value=0.0,
                max_value=100.0
            )
            min_confidence = min_confidence_pct / 100.0
        st.session_state.min_support = min_support
        st.session_state.min_confidence = min_confidence
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

        st.subheader("Definição das meta regras")
        
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
                st.session_state._reset_index = True

        # Criar lista temporária para reconstruir sem o item removido
        novas_regras = []

        # Mostrar regras já adicionadas com opção de remover
        if st.session_state.regras:
            st.markdown(
                "<p style='font-size:18px; font-weight:bold;'>Meta regras selecionadas</p>",
                unsafe_allow_html=True
            )
            
            for i, regra in enumerate(st.session_state.regras):
                texto = f"{regra['antecedente']} → {regra['consequente']}"
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
        
        # =============================
        # Botão para gerar regras filtradas
        # =============================
        if st.button("Gerar regras"):
            # reseta flag para evitar replotagem dupla
            st.session_state.mostrar_regras = False
            progress = st.progress(0)

            st.info("Gerando regras de associação para cada meta-regra...")
            progress.progress(25)

            todas_regras = []

            # 🔹 Gera as regras de forma independente para cada meta-regra
            for meta in st.session_state.regras:
                antecedente = meta["antecedente"]
                consequente = meta["consequente"]

                # Subconjunto apenas com as colunas de interesse
                df_sub = st.session_state.dados_processados[[antecedente, consequente]].copy()

                # Gera regras com mlxtend
                df_temp = analysis.gerar_regras_com_mlxtend(
                    df_sub,
                    sup=st.session_state.min_support,
                    conf=st.session_state.min_confidence
                )

                # Armazena com metadados
                if not df_temp.empty:
                    df_temp["lhs_attr"] = antecedente
                    df_temp["rhs_attr"] = consequente
                    todas_regras.append(df_temp)

            # Junta todas as regras geradas
            if todas_regras:
                df_regras = pd.concat(todas_regras, ignore_index=True)
                st.session_state.df_regras = df_regras
                st.success(f"{len(df_regras)} regras geradas no total.")
            else:
                df_regras = pd.DataFrame()
                st.session_state.df_regras = df_regras
                st.warning("Nenhuma regra foi gerada para as meta-regras selecionadas.")

            progress.progress(50)

            # Etapa 2: filtragem (só salva no estado)
            base_regra = analysis.filtrar_regras_por_atributo(
                df_regras, st.session_state.regras
            )
            st.session_state.base_regra = base_regra
            progress.progress(100)

            # ✅ Marca para renderizar fora do botão
            st.session_state.mostrar_regras = True

            # ✅ Força rerun para que o bloco de gráficos abaixo execute sozinho
            try:
                st.rerun()  # Streamlit >= 1.32
            except Exception:
                st.experimental_rerun()  # compatibilidade com versões anteriores


        # =============================
        # Reexibe gráficos salvos (único bloco que plota)
        # =============================
        if (
            st.session_state.get("mostrar_regras", False)
            and "df_regras" in st.session_state
            and st.session_state.df_regras is not None
            and not st.session_state.df_regras.empty
        ):

            df_regras = st.session_state.df_regras
            base_regra = st.session_state.base_regra

            if not base_regra.empty:
                st.markdown(
                    "<h3 style='text-align: center; font-weight: bold; color: #222;'>Análise Geral das Regras</h3>",
                    unsafe_allow_html=True
                )

                df_plot = base_regra.rename(columns={
                    "Suporte": "suporte",
                    "Confianca": "confianca",
                    "Lift": "lift",
                    "Antecedente": "antecedente",
                    "Consequente": "consequente",
                })

                for regra_user in st.session_state.regras:
                    atributo_antecedente = regra_user["antecedente"]
                    atributo_consequente = regra_user["consequente"]

                    st.markdown(
                        f"<p style='font-size:18px; text-align: center;'>Meta regra: {atributo_antecedente} → {atributo_consequente}</p>",
                        unsafe_allow_html=True
                    )

                    base_regra_filtrada = base_regra[
                        (base_regra["antecedente"].str.contains(atributo_antecedente, case=False, na=False)) &
                        (base_regra["consequente"].str.contains(atributo_consequente, case=False, na=False))
                    ]

                    for cons_val, grupo_cons in base_regra_filtrada.groupby("consequente"):
                        st.markdown(
                            f"<h5 style='text-align:center; color:#222; margin-top:10px; margin-bottom:4px;'>"
                            f"{atributo_antecedente} → {cons_val}</h5>",
                            unsafe_allow_html=True
                        )

                        cols = st.columns(3)
                        cores_fixas = {
                            "suporte": "skyblue",
                            "confianca": "skyblue",
                            "lift": "skyblue"
                        }

                        for i, medida in enumerate(["suporte", "confianca", "lift"]):
                            with cols[i]:
                                fig = px.bar(
                                    grupo_cons,
                                    x="antecedente",
                                    y=medida,
                                    text=grupo_cons[medida].apply(lambda x: f"{x:.2f}"),
                                    title=medida.capitalize(),
                                )
                                fig.update_traces(
                                    marker_color=cores_fixas[medida],
                                    marker_line_color="rgba(0,0,0,0.2)",
                                    marker_line_width=1,
                                    width=0.5,
                                    texttemplate="%{text}",
                                    textposition="outside",
                                    textfont=dict(size=10),
                                    cliponaxis=False
                                )
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=10, r=10, t=40, b=40),
                                    title=dict(text=medida.capitalize(), x=0.5, xanchor='center'),
                                    title_font=dict(size=14, color="#222", family="Arial", weight="normal"),
                                    xaxis=dict(
                                        tickangle=45,
                                        tickfont=dict(size=10),
                                        title=""
                                    ),
                                    yaxis=dict(title=None, tickfont=dict(size=10)),
                                    plot_bgcolor="white",
                                    paper_bgcolor="white",
                                    showlegend=False
                                )
                                # ✅ chave única: adiciona hash do grupo
                                unique_key = f"{atributo_antecedente}_{atributo_consequente}_{cons_val}_{medida}_{hash(cons_val)}"
                                st.plotly_chart(fig, use_container_width=True, key=unique_key)

                st.markdown("<div style='height:1px; background:#e6e6e6; margin:6px 0;'></div>", unsafe_allow_html=True)

    else:
        st.warning("Por favor, carregue o arquivo CSV antes de continuar.")

# ---------- Aba 3: Particionamento da base de dados ----------
elif tab == "Partições":
    st.subheader("Resumo")

    # Verifica se o arquivo original foi carregado
    if st.session_state.dados_original is None:
        st.warning("Por favor, carregue o arquivo CSV antes de continuar.")
    elif not st.session_state.regras:
        st.warning("Nenhuma regra selecionada na aba 2. Selecione ao menos uma regra.")
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
            col1, col2 = st.columns([1, 1])
            with col1:
                # Mostrar a primeira e última data
                data_min = df_original[col_data].min()
                data_max = df_original[col_data].max()
                st.markdown("**Período completo da base de dados**")
                st.write(f"**{data_min.date()}** até **{data_max.date()}**")
            with col2:    
                # Mostrar regras selecionadas
                if st.session_state.regras:
                    st.markdown("**Meta Regras selecionadas para análise**")
                    for r in st.session_state.regras:
                        st.write(f"{r['antecedente']} → {r['consequente']}")
                else:
                    st.info("Nenhuma regra selecionada na aba 2.")

            st.markdown("---")
            
            st.subheader("Tipo de particionamento")
            #Novo seletor de tipo de particionamento
            tipo_particionamento = st.radio(
                "",
                ("Marcos temporais", "Mesmo tamanho", "Mesma quantidade de registros"),
                horizontal=True
            )

            #st.markdown("---")
            #Opção 1: por marcos temporais
            if tipo_particionamento == "Marcos temporais":
                #st.subheader("Defina os marcos temporais")

                # Inicializa lista de marcos no session_state
                if "marcos_temporais" not in st.session_state:
                    st.session_state.marcos_temporais = []

                col1, col2, col3, col4 = st.columns([1,1,1,1])  # col1 menor, col2 maior
                with col1:
                    novo_marco = st.date_input(
                        "Selecione um marco temporal",
                        min_value=data_min.date(),
                        max_value=data_max.date(),
                        label_visibility="collapsed"
                    )
                with col2:
                    # Botão para adicionar marco
                    if st.button("Adicionar marco"):
                        if novo_marco not in st.session_state.marcos_temporais:
                            st.session_state.marcos_temporais.append(novo_marco)
                            #st.success(f"Marco {novo_marco} adicionado!")
                        else:
                            st.warning("Este marco já foi adicionado.")

                # Mostrar lista de marcos já adicionados com opção de remover
                if st.session_state.marcos_temporais:
                    st.write("Marcos temporais definidos:")
                    for i, m in enumerate(sorted(st.session_state.marcos_temporais)):
                        col1, col2, col3, col4 = st.columns([1,1,1,1])
                        with col1:
                            st.write(f"- {m}")
                        with col2:
                            if st.button("❌", key=f"remove_marco_{i}"):
                                st.session_state.marcos_temporais.pop(i)
            
                #st.subheader("Particionar dados por marcos temporais")

                if st.button("Gerar Partições"):
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
                                st.write(f"Partição {i+1}: **{p['inicio'].date()}** até **{p['fim'].date()}** — {len(p['dados'])} registros")
            
            #Opção 2: por janela fixa de tempo (escolhendo a quantidade de tempo)
            elif tipo_particionamento == "Mesmo tamanho":
                #st.subheader("Particionamento por janelas fixas")
                # Input do usuário: quantos meses por partição
                st.markdown("""
                    <style>
                    /* Altera apenas o campo number_input */
                    div[data-testid="stNumberInput"] {
                        width: 130px !important; /* ajuste o valor conforme quiser */
                    }
                    </style>
                    """, unsafe_allow_html=True)
                num_particao = st.number_input(
                    "Número de partições",
                    min_value=1,
                    max_value=60,
                    value=12
                )
                # Botão para gerar as partições
                if st.button("Gerar partições"):
                    particoes_fixas = analysis.particionar_por_tempo_equal_length(df_original, col_data, num_particao)
        
                    st.success(f"Foram geradas {len(particoes_fixas)} partições!")
                    # Salva no session_state
                    st.session_state.particoes_temporais = particoes_fixas
                    # Exibir informações de cada partição em uma linha só
                    for i, p in enumerate(particoes_fixas):
                        delta = relativedelta(p['data_max'], p['data_min'])
                        duracao_texto = []
                        if delta.years > 0:
                            duracao_texto.append(f"{delta.years} ano{'s' if delta.years > 1 else ''}")
                        if delta.months > 0:
                            duracao_texto.append(f"{delta.months} mes{'es' if delta.months > 1 else ''}")
                        if delta.days > 0 and delta.years == 0:
                            duracao_texto.append(f"{delta.days} dia{'s' if delta.days > 1 else ''}")

                        duracao_formatada = ", ".join(duracao_texto) or "0 dias"

                        st.write(
                            f"Partição {i+1}: {len(p['dados'])} registros | "
                            f"{p['data_min'].date()} → {p['data_max'].date()} "
                            f"({duracao_formatada})"
                        )
            
            #Opção 2: por janela fixa de tempo (escolhendo a quantidade de tempo)
            #elif tipo_particionamento == "Por janelas fixas de tempo":
                #st.subheader("Particionamento por janelas fixas")
                # Input do usuário: quantos meses por partição
            #    st.markdown("""
            #        <style>
            #        /* Altera apenas o campo number_input */
            #        div[data-testid="stNumberInput"] {
            #            width: 130px !important; /* ajuste o valor conforme quiser */
            #        }
            #        </style>
            #        """, unsafe_allow_html=True)
            #    meses_por_particao = st.number_input(
            #        "Meses por partição",
            #        min_value=1,
            #        max_value=60,
            #        value=12
            #    )
                # Botão para gerar as partições
                #if st.button("Gerar partições"):
                #    particoes_fixas = analysis.particionar_por_meses(df_original, col_data=col_data, meses_por_particao=meses_por_particao)
        
                #    st.success(f"Foram geradas {len(particoes_fixas)} partições!")
                    # Salva no session_state
                #    st.session_state.particoes_temporais = particoes_fixas
                    # Exibir informações de cada partição em uma linha só
                #    for i, p in enumerate(particoes_fixas):
                #        st.write(f"Partição {i+1}: {len(p['dados'])} registros | {p['data_min'].date()} → {p['data_max'].date()}")

            elif tipo_particionamento == "Mesma quantidade de registros":

                # Estilo igual ao outro particionamento (controla largura do campo)
                st.markdown("""
                    <style>
                    /* Altera apenas o campo number_input */
                    div[data-testid="stNumberInput"] {
                        width: 130px !important; /* mesmo tamanho do outro campo */
                    }
                    </style>
                    """, unsafe_allow_html=True)

                # Campo numérico no mesmo formato
                qtd_particao = st.number_input(
                    "Número de partições:",
                    min_value=1,
                    max_value=len(df_original) if len(df_original) > 0 else 1,
                    value=1,
                    step=1
                )

                qtd_particao = int(qtd_particao)

                # Botão para gerar as partições
                if st.button("Gerar partições"):
                    particoes_registros = analysis.particionar_por_quantidade_igual(df_original, qtd_particao, col_data=col_data)
                    
                    st.success(f"Foram geradas {len(particoes_registros)} partições!")
                    # Salva no session_state
                    st.session_state.particoes_temporais = particoes_registros
                    # Exibir informações de cada partição em uma linha só
                    for i, p in enumerate(particoes_registros):
                       st.write(f"Partição {i+1}: {len(p['dados'])} registros | {p['data_min'].date()} → {p['data_max'].date()}")
                                    
            #elif tipo_particionamento == "Por quantidade de registros":
            
                #st.markdown("**Particionamento por quantidade de registros**")

            #    tamanho_particao = numeric_text_input(
            #        label="Quantidade de registros por partição:",
            #        key="tamanho_particao_input",
            #        value=min(100, len(df_original)),
            #        min_value=10,
            #        max_value=len(df_original),
            #        decimals=0,
            #        width=30
            #    )

            #    tamanho_particao = int(tamanho_particao)

                # Botão para gerar as partições
            #    if st.button("Gerar partições"):
            #        particoes_registros = analysis.particionar_por_registros(df_original, tamanho_particao, col_data=col_data)
        
            #        st.success(f"Foram geradas {len(particoes_registros)} partições!")
                    # Salva no session_state
            #        st.session_state.particoes_temporais = particoes_registros
                    # Exibir informações de cada partição em uma linha só
            #        for i, p in enumerate(particoes_registros):
            #           st.write(f"Partição {i+1}: {len(p['dados'])} registros | {p['data_min'].date()} → {p['data_max'].date()}")
                        
# ---------- Aba 4: Análise temporal ----------
elif tab == "Análise Temporal":
    st.subheader("Análise Temporal das Regras")

    if "particoes_temporais" not in st.session_state:
        st.session_state.particoes_temporais = []

    # Verifica se o arquivo original foi carregado
    if st.session_state.dados_original is None:
        st.warning("Por favor, carregue o arquivo CSV antes de continuar.")
    elif not st.session_state.regras:
        st.warning("Nenhuma regra selecionada na aba 2. Selecione ao menos uma regra.")
    elif not st.session_state.particoes_temporais:
        st.warning("O Particionamento não foi realizado. Particione a base na aba 3.")
    else:
        #st.subheader("Análise Temporal das Regras")

        if "particoes_temporais" not in st.session_state or not st.session_state.particoes_temporais:
            st.warning("Por favor, particione os dados antes de gerar a análise temporal.")
        elif not st.session_state.regras:
            st.warning("Nenhuma regra selecionada para análise temporal.")
        else:
            if st.button("Gerar Análise Temporal"):
                resultados = []
                col_data = None

                # Detecta a coluna de data
                for c in st.session_state.particoes_temporais[0]["dados"].columns:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.particoes_temporais[0]["dados"][c]):
                        col_data = c
                        break

                # === Análise Temporal das Regras ===
                #st.subheader("Análise Temporal das Regras")

                # Para cada meta-regra selecionada pelo usuário
                for regra_user in st.session_state.regras:
                    ant_attr = regra_user["antecedente"]
                    cons_attr = regra_user["consequente"]

                    # Filtra base geral para regras que correspondem à meta-regra (antecedente)
                    base_geral_filtrada = st.session_state.base_regra[
                        st.session_state.base_regra["antecedente"].str.match(f"^{ant_attr}=.+$")
                    ]

                    if base_geral_filtrada.empty:
                        st.warning(f"Nenhuma regra encontrada na base geral para {ant_attr} → {cons_attr}")
                        continue

                    # Itera sobre cada valor distinto do antecedente
                    for ant_val, grupo_ant in base_geral_filtrada.groupby("antecedente"):

                        # Para cada valor distinto do consequente dentro do antecedente
                        for cons_val, grupo_cons_geral in grupo_ant.groupby("consequente"):

                            st.markdown(
                                f"<h5 style='text-align:center; color:#222; margin-top:10px; margin-bottom:4px;'>"
                                f"{ant_val} → {cons_val}</h5>",
                                unsafe_allow_html=True
                            )

                            # Lista para armazenar medidas de cada partição
                            medidas_particoes = []

                            # Percorre todas as partições temporais
                            for i, part in enumerate(st.session_state.particoes_temporais):
                                df_part = part["dados"].copy()
                                if df_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0})
                                    continue

                                if "data" in df_part.columns:  # remove coluna de data
                                    df_part = df_part.drop(columns=["data"])

                                # Prepara os dados para mineração
                                df_part_tratado, _, _ = analysis.preparar_dados_para_mineracao_from_df(df_part)

                                # Gera regras na partição
                                df_regras_part = analysis.gerar_regras_com_mlxtend2(
                                    df_part_tratado,
                                    st.session_state.min_support,
                                    st.session_state.min_confidence
                                )

                                if df_regras_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0})
                                    continue

                                df_filtrado_part = df_regras_part[
                                    (df_regras_part["antecedente"] == ant_val) &
                                    (df_regras_part["consequente"] == cons_val)
                                ]

                                if df_filtrado_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0})
                                else:
                                    row = df_filtrado_part.iloc[0]  # assume única correspondência
                                    medidas_particoes.append({
                                        "suporte": row["suporte"],
                                        "confianca": row["confianca"],
                                        "lift": row["lift"]
                                    })

                            # Cria DataFrame com medidas de todas as partições
                            df_medidas = pd.DataFrame(medidas_particoes)
                            df_medidas.index = [f"Partição {i+1}" for i in range(len(medidas_particoes))]

                            # Valores gerais da base geral (linha vermelha)
                            linha_geral = grupo_cons_geral.iloc[0]  # pega a regra completa
                            valores_gerais = {
                                "suporte": linha_geral["suporte"],
                                "confianca": linha_geral["confianca"],
                                "lift": linha_geral["lift"]
                            }

                            # 3 gráficos lado a lado: suporte, confiança, lift
                            cols = st.columns(3)
                            for j, medida in enumerate(["suporte", "confianca", "lift"]):
                                with cols[j]:
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    ax.bar(df_medidas.index, df_medidas[medida])
                                    ax.set_title(medida.capitalize(), fontsize=10, pad=2)
                                    ax.set_ylim(0, max(1, df_medidas[medida].max() * 1.15))
                                    ax.tick_params(axis="x", rotation=45, labelsize=8)
                                    ax.tick_params(axis="y", labelsize=8)

                                    # linha de referência da análise geral
                                    y_ref = valores_gerais[medida]
                                    ax.axhline(y=y_ref, color="red", linestyle="--")
                                    ax.text(len(df_medidas.index)-0.3, y_ref, f"{y_ref:.2f}", color="red",
                                            fontsize=8, va="bottom", ha="left")

                                    # valores em cima das barras
                                    for k, h in enumerate(df_medidas[medida]):
                                        ax.text(k, h, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

                                    plt.tight_layout()
                                    st.pyplot(fig, use_container_width=False)

