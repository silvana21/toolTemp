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
import copy

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

def _sanitize_input(key, min_value=None, max_value=None, decimals=4):
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

def numeric_text_input(label, key, value=0.0, min_value=None, max_value=None, decimals=4, width=90):
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
    st.markdown("### Navegação")

    tab = option_menu(
        None,
        ["Carregar CSV", "Regras Gerais", "Análise Temporal", "Histórico de Análises"],
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

            # Cria três colunas e usa só a central (para não ocupar toda a tela)
            col1, col2, col3 = st.columns([0.3, 0.2, 0.2])

            with col1:  # tudo acontece dentro da primeira coluna
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
        st.subheader("Resumo dos Atributos (valores e frequência)")

        cols = st.columns(2)
        col_index = 0

        for col, value_counts in st.session_state["resumo_colunas"].items():
            ordem_escolhida = st.session_state["ordem_valores"].get(
                col, value_counts.index.tolist()
            )
            # aplica ordem escolhida
            value_counts = value_counts.reindex(ordem_escolhida)

            df_temp = value_counts.reset_index()
            df_temp.columns = ["valor", "frequencia"]
            df_temp["atributo"] = col  # adiciona o nome do atributo para o hover

            # ===== Gráfico de barras horizontais =====
            fig = px.bar(
                df_temp,
                y="valor",
                x="frequencia",
                text="frequencia",
                color_discrete_sequence=["#4A90E2"],  # tom padrão
                height=220,
                orientation="h"  # 👈 barra horizontal
            )

            # ===== Hover customizado =====
            fig.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
                customdata=df_temp[["atributo"]],
                hovertemplate=(
                    "Atributo: %{customdata[0]}<br>"
                    + "Valor: %{y}<br>"
                    + "Quantidade: %{x}<extra></extra>"
                ),
            )

            # ===== Layout =====
            fig.update_layout(
                title_text="",
                xaxis_title="",
                yaxis_title="",
                margin=dict(l=80, r=20, t=10, b=10),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(tickfont=dict(size=11), automargin=True)
            fig.update_xaxes(showgrid=False, tickfont=dict(size=11))

            # ===== Exibição =====
            with cols[col_index]:
                st.markdown(
                    f"<p style='text-align: center; font-weight: bold;'>{col}</p>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            col_index += 1
            if col_index == 2:
                st.markdown(
                    "<div style='height:1px; background-color:#e0e0e0; margin:5px 0;'></div>",
                    unsafe_allow_html=True
                )
                cols = st.columns(2)
                col_index = 0

            total_valores = df_proc[col].nunique(dropna=False)
            if total_valores > 10:
                st.info(f"Atributo `{col}` possui {total_valores} valores. Exibindo apenas os 10 mais frequentes.")

    # 3) Caso não tenha nada ainda
    else:
        st.warning("Nenhum arquivo CSV foi carregado ainda.")

# --- ABA 2: Definição de Regras ---
elif tab == "Regras Gerais":
    if "__rerun_trigger__" in st.session_state:
        del st.session_state["__rerun_trigger__"]
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

        # ===============================
        # Inicialização de variáveis
        # ===============================
        if "regras" not in st.session_state:
            st.session_state.regras = []
        if "antecedente_habilitado" not in st.session_state:
            st.session_state.antecedente_habilitado = True
        if "consequente_habilitado" not in st.session_state:
            st.session_state.consequente_habilitado = True
        if "antecedente_select" not in st.session_state:
            st.session_state.antecedente_select = []
        if "consequente_select" not in st.session_state:
            st.session_state.consequente_select = ""
        if "reset_flag" not in st.session_state:
            st.session_state.reset_flag = False

        # ===============================
        # Reset seguro após rerun
        # ===============================
        if st.session_state.reset_flag:
            st.session_state.antecedente_select = []
            st.session_state.consequente_select = ""
            st.session_state.reset_flag = False

        # ===============================
        # Interface de meta-regras
        # ===============================
        st.subheader("Definição das meta regras")

        col_e, col_cent, col_d = st.columns([2, 1, 1])
        with col_e:
            col_1, col_2 = st.columns([1, 1])

        colunas_lista = colunas if "colunas" in locals() else st.session_state.dados_processados.columns.tolist()

        # ==========================
        # ANTECEDENTE(S)
        # ==========================

        # inicialização segura no estado
        if "antecedente_select" not in st.session_state or st.session_state.antecedente_select is None:
            st.session_state.antecedente_select = []

        # converte string para lista, se necessário
        if isinstance(st.session_state.antecedente_select, str):
            st.session_state.antecedente_select = (
                [] if not st.session_state.antecedente_select.strip()
                else [st.session_state.antecedente_select.strip()]
            )

        with col_1:
            # a chave e o valor agora são controlados **somente pelo estado**
            antecedentes = st.multiselect(
                "Selecione o(s) Antecedente(s)",
                options=colunas_lista,
                key="antecedente_select",
                disabled=not st.session_state.antecedente_habilitado,
                help="Você pode selecionar mais de um atributo para criar uma meta-regra composta."
            )

        # --- CONSEQUENTE (remove os escolhidos como antecedente) ---
        opcoes_consequente = [c for c in colunas_lista if c not in antecedentes]
        with col_2:
            consequente = st.selectbox(
                "Selecione o Consequente",
                options=[""] + opcoes_consequente,
                index=0 if st.session_state.consequente_select not in opcoes_consequente
                else opcoes_consequente.index(st.session_state.consequente_select) + 1,
                key="consequente_select",
                disabled=not st.session_state.consequente_habilitado
            )

        # --- Botão para adicionar meta-regra ---
        if st.button("Adicionar meta regra"):
            if antecedentes and consequente:
                regra = {"antecedente": ", ".join(antecedentes), "consequente": consequente}
                if regra not in st.session_state.regras:
                    st.session_state.regras.append(regra)
                st.session_state.reset_flag = True
                st.rerun()

        # ===============================
        # Exibe regras adicionadas
        # ===============================
        novas_regras = []
        if st.session_state.regras:
            st.markdown(
                "<p style='font-size:18px; font-weight:bold;'>Meta regras selecionadas</p>",
                unsafe_allow_html=True
            )
            for i, regra in enumerate(st.session_state.regras):
                texto = f"{regra['antecedente']} → {regra['consequente']}"
                col1, col2 = st.columns([4, 1])
                remover = False
                with col1:
                    st.write(texto)
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        remover = True
                if not remover:
                    novas_regras.append(regra)

        st.session_state.regras = novas_regras

        # ===============================
        # Início da contagem total
        # ===============================
        inicio_total = time.time()

        # =============================
        # Botão para gerar regras filtradas
        # =============================
        if st.button("Gerar regras"):
            st.session_state.mostrar_regras = False
            progress = st.progress(0)
            
            st.info("Gerando regras de associação para cada meta-regra...")
            inicio_regras = time.time()
            progress.progress(25)

            todas_regras = []

            for meta in st.session_state.regras:
                antecedente = meta["antecedente"]
                consequente = meta["consequente"]

                ant_cols = analysis._parse_antecedentes(antecedente)

                missing = [c for c in ant_cols + [consequente] 
                        if c not in st.session_state.dados_processados.columns]
                if missing:
                    st.error(f"A(s) coluna(s) {missing} não existem no dataset.")
                    st.stop()

                df_sub = st.session_state.dados_processados[ant_cols + [consequente]].copy()

                df_temp = analysis.gerar_regras_com_mlxtend(
                    df_sub,
                    sup=st.session_state.min_support,
                    conf=st.session_state.min_confidence
                )

                if not df_temp.empty:
                    df_temp["lhs_attr"] = antecedente
                    df_temp["rhs_attr"] = consequente
                    todas_regras.append(df_temp)

            fim_regras = time.time()
            tempo_regras = fim_regras - inicio_regras

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

            # =============================
            # Etapa 2: Filtragem
            # =============================
            st.info("🔎 Filtrando regras conforme meta-regras...")
            inicio_filtro = time.time()

            base_regra = analysis.filtrar_regras_por_atributo(
                df_regras, st.session_state.regras
            )
            
            fim_filtro = time.time()
            tempo_filtro = fim_filtro - inicio_filtro

            st.session_state.base_regra = base_regra
            st.session_state.df_regras = df_regras
            st.session_state.mostrar_regras = True

            progress.progress(100)
            st.session_state["tempos_execucao"] = {
                "tempo_regras": tempo_regras,
                "tempo_filtro": tempo_filtro,
            }

            st.session_state["__rerun_trigger__"] = True  
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

        # =============================
        # Reexibe gráficos salvos
        # =============================
        
        inicio_graficos = time.time()

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

                for regra_user in st.session_state.regras:
                    atributo_antecedente = regra_user["antecedente"]
                    atributo_consequente = regra_user["consequente"]
                    meta_tag = f"{atributo_antecedente} → {atributo_consequente}"

                    st.markdown(
                        f"<p style='font-size:18px; text-align: center;'>Meta regra: {meta_tag}</p>",
                        unsafe_allow_html=True
                    )

                    if "meta_regra" in base_regra.columns:
                        base_regra_filtrada = base_regra[base_regra["meta_regra"] == meta_tag].copy()
                    else:
                        base_regra_filtrada = pd.DataFrame()

                    base_regra_filtrada = base_regra_filtrada.drop_duplicates(
                        subset=["antecedente", "consequente", "suporte", "confianca", "lift"]
                    )

                    for cons_val, grupo_cons in base_regra_filtrada.groupby("consequente"):
                        st.markdown(
                            f"<h5 style='text-align:center; color:#222; margin-top:10px; margin-bottom:4px;'>"
                            f"{atributo_antecedente} → {cons_val}</h5>",
                            unsafe_allow_html=True
                        )

                        grupo_cons = grupo_cons.copy()

                        # === aplica ordem salva na aba 1, se existir === 
                        if atributo_antecedente in st.session_state.get("ordem_valores", {}): 
                            ordem = st.session_state["ordem_valores"][atributo_antecedente] # detecta formato (com ou sem "atributo=") 
                            exemplo = str(grupo_cons["antecedente"].iloc[0]) 
                            if "=" in exemplo and exemplo.split("=")[0].strip().lower() == atributo_antecedente.lower(): 
                                ordem_labels = [f"{atributo_antecedente}={v}" for v in ordem] 
                            else: ordem_labels = ordem 
                            
                            grupo_cons["antecedente"] = pd.Categorical( 
                                grupo_cons["antecedente"].astype(str), 
                                categories=ordem_labels, ordered=True 
                            ) 
                            grupo_cons = grupo_cons.sort_values("antecedente") 
                        else: 
                            ordem_labels = None

                        cols = st.columns(3)
                        cores_fixas = {"suporte": "skyblue", "confianca": "skyblue", "lift": "skyblue"}

                        for i, medida in enumerate(["suporte", "confianca", "lift"]):
                            with cols[i]:
                                fig = px.bar(
                                    grupo_cons,
                                    x="antecedente",
                                    y=medida,
                                    text=grupo_cons[medida].apply(lambda x: f"{x:.2f}"),
                                    title=medida.capitalize(),
                                    category_orders={"antecedente": ordem_labels} if ordem_labels else None
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
                                    xaxis=dict(tickangle=45, tickfont=dict(size=10), title="", showgrid=False), 
                                    yaxis=dict(title=None, tickfont=dict(size=10), showgrid=False), 
                                    plot_bgcolor="white", 
                                    paper_bgcolor="white", 
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)

                fim_graficos = time.time()
                tempo_graficos = fim_graficos - inicio_graficos

                tempos = st.session_state.get("tempos_execucao", {})
                tempo_regras = tempos.get("tempo_regras", 0)
                tempo_filtro = tempos.get("tempo_filtro", 0)

                tempo_total = tempo_regras + tempo_filtro + tempo_graficos  # soma explícita

                st.markdown("---")
                st.info(f"Tempo para gerar regras: **{tempo_regras:.2f} s**")
                st.info(f"Tempo para filtrar regras: **{tempo_filtro:.2f} s**")
                st.info(f"Tempo para gerar gráficos: **{tempo_graficos:.2f} s**")
                st.success(f"Tempo total: **{tempo_total:.2f} s**")

    else:
        st.warning("Por favor, carregue o arquivo CSV antes de continuar.")


# ---------- Aba 3: Particionamento da base de dados ----------
elif tab == "Análise Temporal":
    st.subheader("Resumo")
    

    # ==============================================
    # Inicializa estrutura de abas de análises, se necessário
    # ==============================================
    if "analises_temporais" not in st.session_state:
        st.session_state.analises_temporais = []
    
    
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
            
            #st.subheader("Tipo de particionamento")
            #Novo seletor de tipo de particionamento
            #tipo_particionamento = st.radio(
            #    "",
            #    ("Marcos temporais", "Mesmo tamanho", "Mesma quantidade de registros"),
            #    horizontal=True
            #)
            # 🔹 Guarda o tipo de particionamento escolhido para reutilizar depois
            #st.session_state.tipo_particionamento = tipo_particionamento

            #st.markdown("---")
            # ======================================
            # SEÇÃO DE PARTICIONAMENTO REESTILIZADA
            # ======================================

            st.markdown("### Particionamento da Base de Dados")

            col_left, col_right = st.columns([1.5, 2.5])

            # --- COLUNA ESQUERDA: tipo de particionamento ---
            with col_left:
                tipo_particionamento = st.radio(
                    "Selecione o tipo:",
                    ("Marcos temporais", "Mesmo tamanho temporal", "Mesma quantidade de registros"),
                    index=0,
                    key="tipo_particionamento_radio",
                )

            # --- COLUNA DIREITA: conteúdo dinâmico conforme o tipo ---
            with col_right:
                # Opção 1: Marcos temporais
                if tipo_particionamento == "Marcos temporais":
                    if "marcos_temporais" not in st.session_state:
                        st.session_state.marcos_temporais = []

                    novo_marco = st.date_input(
                        "Selecione um marco temporal",
                        min_value=data_min.date(),
                        max_value=data_max.date(),
                    )

                    col_add, col_list = st.columns([1, 3])
                    with col_add:
                        if st.button("Adicionar marco", key="add_marco_btn"):
                            if novo_marco not in st.session_state.marcos_temporais:
                                st.session_state.marcos_temporais.append(novo_marco)
                            else:
                                st.warning("Este marco já foi adicionado.")
                    
                    with col_list:
                        if st.session_state.marcos_temporais:
                            st.write("Marcos definidos:")
                            for i, m in enumerate(sorted(st.session_state.marcos_temporais)):
                                col_m1, col_m2 = st.columns([4, 1])
                                with col_m1:
                                    st.write(f"- {m}")
                                with col_m2:
                                    if st.button("❌", key=f"remove_marco_{i}"):
                                        st.session_state.marcos_temporais.pop(i)

                    col_btn, col_msg = st.columns([1, 3])
                    with col_btn:
                        gerar_marcos = st.button("Gerar Partições", key="btn_particoes_marcos")

                    if gerar_marcos:
                        if not st.session_state.marcos_temporais:
                            st.warning("Nenhum marco temporal definido.")
                        else:
                            df_original = st.session_state.dados_original.copy()
                            col_data = None
                            for c in df_original.columns:
                                if pd.api.types.is_datetime64_any_dtype(df_original[c]):
                                    col_data = c
                                    break
                            if col_data:
                                marcos = [pd.to_datetime(m) for m in sorted(st.session_state.marcos_temporais)]
                                data_min = df_original[col_data].min()
                                data_max = df_original[col_data].max()
                                limites = [data_min] + marcos + [data_max]
                                particoes = []
                                for i in range(len(limites) - 1):
                                    inicio = limites[i]
                                    fim = limites[i + 1]
                                    if i > 0:
                                        inicio += pd.Timedelta(days=1)
                                    part = df_original[
                                        (df_original[col_data] >= inicio)
                                        & (df_original[col_data] <= fim)
                                    ].copy()
                                    particoes.append({"inicio": inicio, "fim": fim, "dados": part})

                                st.session_state.particoes_temporais = particoes
                                st.session_state.analise_temporal_em_andamento = False
                                st.session_state.analise_temporal_pronta = False
                                st.session_state.particoes_last_msg = (
                                    f"Particionamento concluído! Total de partições: {len(particoes)}"
                                )

                    with col_msg:
                        if "particoes_last_msg" in st.session_state:
                            st.success(st.session_state.particoes_last_msg)

                # Opção 2: Mesmo tamanho
                elif tipo_particionamento == "Mesmo tamanho temporal":

                    # 🔧 CSS: deixa o number_input mais compacto, MAS preservando os botões +/-
                    st.markdown("""
                        <style>
                        /* A caixa externa continua grande o suficiente para os botões */
                        div[data-testid="stNumberInput"] > div {
                            max-width: 110px !important;    /* largura interna do input */
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # 🔹 Organização em linha: número | botão | mensagem
                    col_num, col_btn, col_msg = st.columns([2, 1, 2])

                    with col_num:
                        # 🔹 Subdivide a coluna: rótulo | input
                        c_rotulo, c_input = st.columns([1.5, 2])

                        with c_rotulo:
                            st.markdown("Num. partições: ")

                        with c_input:
                            num_particao = st.number_input(
                                label="N. partições: ",
                                min_value=1,
                                max_value=60,
                                value=1,
                                key="num_particoes_tempo",
                                label_visibility="collapsed",  # escondemos o label nativo
                            )

                    with col_btn:
                        gerar_fixas = st.button(
                            "Gerar Partições",
                            key="btn_particoes_tempo",
                            use_container_width=True
                        )

                    if gerar_fixas:
                        particoes_fixas = analysis.particionar_por_tempo_equal_length(
                            df_original, col_data, num_particao
                        )

                        st.session_state.particoes_temporais = particoes_fixas
                        st.session_state.analise_temporal_em_andamento = False
                        st.session_state.analise_temporal_pronta = False
                        st.session_state.analise_atual = None

                        st.session_state.particoes_last_msg = (
                            f"{len(particoes_fixas)} partições geradas!"
                        )

                    with col_msg:
                        if "particoes_last_msg" in st.session_state:
                            st.success(f"✅ {st.session_state.particoes_last_msg}")


                # Opção 3: Mesma quantidade de registros
                elif tipo_particionamento == "Mesma quantidade de registros":
                    qtd_particao = st.number_input(
                        "Número de partições:",
                        min_value=1,
                        max_value=len(df_original) if len(df_original) > 0 else 1,
                        value=1,
                        step=1
                    )

                    col_btn, col_msg = st.columns([2, 2])
                    with col_btn:
                        gerar_qtd = st.button("Gerar Partições", key="btn_particoes_qtd")

                    if gerar_qtd:
                        particoes_registros = analysis.particionar_por_quantidade_igual(
                            df_original, int(qtd_particao), col_data=col_data
                        )
                        st.session_state.particoes_temporais = particoes_registros
                        st.session_state.analise_temporal_em_andamento = False
                        st.session_state.analise_temporal_pronta = False
                        st.session_state.particoes_last_msg = (
                            f"{len(particoes_registros)} partições geradas!"
                        )

                    with col_msg:
                        if "particoes_last_msg" in st.session_state:
                            st.success(st.session_state.particoes_last_msg)


            # === Resumo persistente das partições + botão da análise temporal ===
            if "particoes_temporais" in st.session_state and st.session_state.particoes_temporais:
                #st.success(st.session_state.get("particoes_last_msg", f"{len(st.session_state.particoes_temporais)} partições geradas."))

                # Renderização do resumo (UM lugar só)
                for i, p in enumerate(st.session_state.particoes_temporais):
                    # Detecta se é "Mesmo tamanho" com base no tipo salvo
                    tipo = st.session_state.get("tipo_particionamento", "")

                    # Suporta ambos formatos (inicio/fim ou data_min/data_max)
                    if "inicio" in p and "fim" in p:
                        data_ini, data_fim = p["inicio"], p["fim"]
                    else:
                        data_ini, data_fim = p["data_min"], p["data_max"]

                    duracao_texto = ""
                    #if tipo == "Mesmo tamanho temporal":
                    delta = relativedelta(data_fim, data_ini)
                    partes = []
                    if delta.years > 0:
                        partes.append(f"{delta.years} ano{'s' if delta.years > 1 else ''}")
                    if delta.months > 0:
                        partes.append(f"{delta.months} mes{'es' if delta.months > 1 else ''}")
                    if delta.days > 0 and not partes:
                        partes.append(f"{delta.days} dia{'s' if delta.days > 1 else ''}")
                    
                    duracao_calculada = ", ".join(partes)
                    duracao_texto = f" ({duracao_calculada if duracao_calculada else '0 dias'})"

                    st.write(
                        f"Intervalo {i+1}: **{data_ini.strftime('%d/%m/%Y')}** → **{data_fim.strftime('%d/%m/%Y')}** — "
                        f"{len(p['dados'])} registros {duracao_texto}"
                    )

                st.markdown("---")
                if st.button("Gerar Análise Temporal", key="botao_analise_temporal"):
                    # 🔹 Garante que a análise só será iniciada manualmente
                    st.session_state.analise_temporal_em_andamento = True
                    st.session_state.analise_temporal_pronta = False
                    st.session_state.analise_atual = None

            # --- BLOCO SEPARADO: EXECUTA ANÁLISE TEMPORAL QUANDO A FLAG ESTIVER ATIVA ---
            if st.session_state.get("analise_temporal_em_andamento", False) and not st.session_state.get("analise_temporal_pronta", False):

                st.info("🔍 Iniciando análise temporal das regras...")

                resultados = []
                col_data = None

                # Detecta a coluna de data
                for c in st.session_state.particoes_temporais[0]["dados"].columns:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.particoes_temporais[0]["dados"][c]):
                        col_data = c
                        break
                
                # ==========================================================
                # Lista para guardar tudo o que será gerado nesta análise
                # ==========================================================
                analise_atual = {
                    "graficos": [],
                    "tabelas": [],
                    "descricao": f"Análise com {len(st.session_state.particoes_temporais)} partições e {len(st.session_state.regras)} meta-regra(s)"
                }


                # === Análise Temporal das Regras ===
                for regra_user in st.session_state.regras:
                    ant_attr = regra_user["antecedente"]
                    cons_attr = regra_user["consequente"]

                    base_geral_filtrada = st.session_state.base_regra[
                        st.session_state.base_regra["antecedente"].str.match(f"^{ant_attr}=.+$")
                    ]

                    if base_geral_filtrada.empty:
                        st.warning(f"Nenhuma regra encontrada na base geral para {ant_attr} → {cons_attr}")
                        continue

                    for ant_val, grupo_ant in base_geral_filtrada.groupby("antecedente"):
                        for cons_val, grupo_cons_geral in grupo_ant.groupby("consequente"):
                            st.markdown(
                                f"<h5 style='text-align:center; color:#222; margin-top:10px; margin-bottom:4px;'>"
                                f"{ant_val} → {cons_val}</h5>",
                                unsafe_allow_html=True
                            )

                            medidas_particoes = []

                            for i, part in enumerate(st.session_state.particoes_temporais):
                                df_part = part["dados"].copy()
                                if df_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0})
                                    continue

                                if "data" in df_part.columns:
                                    df_part = df_part.drop(columns=["data"])

                                df_part_tratado, _, _ = analysis.preparar_dados_para_mineracao_from_df(df_part)
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
                                    row = df_filtrado_part.iloc[0]
                                    medidas_particoes.append({
                                        "suporte": row["suporte"],
                                        "confianca": row["confianca"],
                                        "lift": row["lift"]
                                    })

                            df_medidas = pd.DataFrame(medidas_particoes)
                            df_medidas.index = [f"Partição {i+1}" for i in range(len(medidas_particoes))]

                            linha_geral = grupo_cons_geral.iloc[0]
                            valores_gerais = {
                                "suporte": linha_geral["suporte"],
                                "confianca": linha_geral["confianca"],
                                "lift": linha_geral["lift"]
                            }

                            cores_fixas = {
                                "suporte": "skyblue",      
                                "confianca": "skyblue",    
                                "lift": "skyblue"          
                            }
                            # Monta os períodos de cada partição (ex: "01/01/2015 → 31/12/2015")
                            periodos = []
                            for p in st.session_state.particoes_temporais:
                                if "inicio" in p and "fim" in p:
                                    ini, fim = p["inicio"].date(), p["fim"].date()
                                else:
                                    ini, fim = p["data_min"].date(), p["data_max"].date()
                                periodos.append(f"{ini.strftime('%d/%m/%Y')} → {fim.strftime('%d/%m/%Y')}")
                            df_medidas["Periodo"] = periodos
                            cols = st.columns(3)
                            figs_regra = {}
                            for j, medida in enumerate(["suporte", "confianca", "lift"]):
                                with cols[j]:
                                    
                                    # Cria gráfico de barras com valores da medida
                                    fig = px.bar(
                                        df_medidas,
                                        #x=df_medidas.index,
                                        x="Periodo",
                                        y=medida,
                                        text=df_medidas[medida].apply(lambda x: f"{x:.2f}"),
                                        title=medida.capitalize(),
                                    )

                                    # Linha de referência geral (suave)
                                    y_ref = valores_gerais[medida]
                                    fig.add_hline(
                                        y=y_ref,
                                        line_dash="dot",
                                        line_color="red",
                                        line_width=1,
                                        opacity=0.4
                                    )

                                    # Pega o índice da última barra para posicionar o texto logo depois
                                    x_final = len(df_medidas.index) - 0.4  # ajusta o deslocamento horizontal

                                    # Adiciona o valor da linha ao final dela
                                    fig.add_annotation(
                                        x=x_final,                # 🔹 depois da última barra
                                        y=y_ref,                  # mesma altura da linha
                                        text=f"{y_ref:.2f}",      # valor formatado
                                        showarrow=False,
                                        font=dict(color="red", size=10),
                                        xanchor="left",           # 🔹 texto “depois” da linha
                                        yanchor="bottom",
                                        xshift=10                 # 🔹 desloca levemente para a direita
                                    )


                                    # Estilo das barras
                                    fig.update_traces(
                                        marker_color=cores_fixas[medida],
                                        texttemplate="%{text}",
                                        textposition="outside",
                                        textfont=dict(size=10),
                                        cliponaxis=False,
                                        hovertemplate=(
                                            " <b>%{customdata[0]}</b><br>"  # mostra o período da partição
                                            f"{medida.capitalize()}: <b>%{{y:.2f}}</b><extra></extra>"
                                        ),
                                        customdata=df_medidas[["Periodo"]]  # 🔹 adiciona a coluna extra usada no hover
                                    )

                                    # Layout e formatação geral
                                    fig.update_layout(
                                        height=350,
                                        margin=dict(l=10, r=10, t=50, b=50),
                                        title=dict(
                                            text=medida.capitalize(),
                                            x=0.5,  # 🔹 Centraliza título
                                            xanchor="center",
                                            font=dict(size=14, color="#333", family="Arial", weight="normal")
                                        ),
                                        xaxis=dict(
                                            tickangle=45,
                                            title=None,
                                            tickfont=dict(size=10),
                                            showgrid=False   # ❌ remove grade vertical
                                            #zeroline=False    # ❌ remove linha zero
                                        ),
                                        yaxis=dict(
                                            title=None,
                                            tickfont=dict(size=10),
                                            showgrid=False   # ❌ remove grade horizontal
                                            #zeroline=False    # ❌ remove linha zero
                                        ),
                                        plot_bgcolor="white",
                                        paper_bgcolor="white",
                                        showlegend=False
                                    )

                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"{ant_val}_{cons_val}_{medida}_{i}"
                                    )
                                    
                                    # gera fig normalmente...
                                    figs_regra[medida] = fig
                                    # Guarda um conjunto de 3 gráficos (um grupo por regra)
                                    if "regras_plotadas" not in locals():
                                        regras_plotadas = 0
                                    if j == 2:  # quando chegar no último gráfico (lift)
                                        analise_atual["graficos"].append({
                                            "titulo": f"{ant_val} → {cons_val}",
                                            "figs": figs_regra  # conjunto dos três gráficos
                                        })
                                        regras_plotadas += 1

                nome_analise = f"Análise {len(st.session_state.analises_temporais) + 1}"
                analise_atual["nome"] = nome_analise
                analise_atual["parametros"] = {
                    "suporte_min": st.session_state.min_support,
                    "confianca_min": st.session_state.min_confidence,
                    "meta_regras": copy.deepcopy(st.session_state.regras),
                    "particoes": copy.deepcopy(st.session_state.particoes_temporais),
                    "tipo_particionamento": st.session_state.get("tipo_particionamento", "Não informado")
                }

                st.session_state.analises_temporais.append(copy.deepcopy(analise_atual))
                # 🔹 Marca como concluída
                st.session_state.analise_temporal_em_andamento = False
                st.session_state.analise_temporal_pronta = True
                st.success(f"💾 {nome_analise} salva com sucesso!")

elif tab == "Histórico de Análises":
    st.subheader("Histórico de Análises Temporais")

    if "analises_temporais" not in st.session_state or not st.session_state.analises_temporais:
        st.info("Nenhuma análise temporal foi realizada ainda.")
    else:
        abas_hist = st.tabs([a["nome"] for a in st.session_state.analises_temporais])

        for i, analise in enumerate(st.session_state.analises_temporais):
            with abas_hist[i]:
                st.markdown(f"### {analise['nome']}")
                st.write(analise["descricao"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Suporte mínimo:** {analise['parametros']['suporte_min']}")
                with col2:
                    st.markdown(f"**Confiança mínima:** {analise['parametros']['confianca_min']}")
                with col3:
                    st.markdown(f"**Tipo de particionamento:** {analise['parametros'].get('tipo_particionamento', '-')}")
                st.markdown("---")

                st.markdown("**Meta-regras analisadas:**")
                for regra in analise['parametros']['meta_regras']:
                    st.write(f"• {regra['antecedente']} → {regra['consequente']}")

                st.markdown("---")
                st.markdown("**Partições geradas:**")
                #duracao_calculada = ", ".join(partes)
                #duracao_texto = f" ({duracao_calculada if duracao_calculada else '0 dias'})"
                for j, p in enumerate(analise['parametros']['particoes']):
                    if "inicio" in p and "fim" in p:
                        st.write(f"Intervalo {j+1}: {p['inicio'].date()} → {p['fim'].date()} — {len(p['dados'])} registros")
                    else:
                        st.write(f"Intervalo {j+1}: {p['data_min'].date()} → {p['data_max'].date()} — {len(p['dados'])} registros")

                st.markdown("---")

                for regra_plotada in analise["graficos"]:
                    st.markdown(f"### {regra_plotada['titulo']}")
                    cols = st.columns(3)
                    for j, medida in enumerate(["suporte", "confianca", "lift"]):
                        with cols[j]:
                            st.plotly_chart(
                                regra_plotada["figs"][medida],
                                use_container_width=True,
                                key=f"hist_{i}_{regra_plotada['titulo']}_{medida}"
                            )
