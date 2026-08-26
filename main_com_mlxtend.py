# main.py
import streamlit as st
import pandas as pd
import analysis  
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import re
from dateutil.relativedelta import relativedelta
import time
import plotly.express as px
import copy
from decimal import Decimal, ROUND_HALF_UP
from datetime import timedelta

st.set_page_config(page_title="Análise Temporal de Regras de Associação", layout="wide")

# ---------- Session state ----------
def _init_state():
    defaults = {
        "dados_original": None,
        "dados_processados": None,
        "regras_df": None,
        "min_support": 0.01,
        "min_confidence": 0.01,
        "min_support_temporal": 0.01,
        "min_confidence_temporal": 0.01,
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

def round2(x):
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

# ----------------- Abas -----------------
# --- MENU LATERAL ---
with st.sidebar:
    st.markdown("### Navegação")

    tab = option_menu(
        None,
        ["Carregar CSV", "Regras Gerais", "Análise Temporal", "Histórico de Análises Gerais", "Histórico de Análises Temporais"],
        icons=["file-earmark-arrow-up", "diagram-3", "calendar3", "bar-chart", "clock-history"],
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
            
            # ======================================================
            # Limpa somente a análise atual caso o outro arquivo seja carregado
            # ---------- Regras Gerais ----------
            for chave in [
                "regras",
                "df_regras",
                "base_regra",
                "mostrar_regras",
                "antecedente_select",
                "consequente_select",
                "reset_flag",
                "salvou_analise_geral",
            ]:
                st.session_state.pop(chave, None)
            # ---------- Análise Temporal ----------
            for chave in [
                "particoes_temporais",
                "particoes_last_msg",
                "marcos_temporais",
                "tipo_particionamento",
                "analise_temporal_em_andamento",
                "analise_temporal_pronta",
                "analise_atual",
                "regras_temporais_mineradas",
            ]:
                st.session_state.pop(chave, None)

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

        # =====================================
        # Recupera coluna temporal do dataset original
        # =====================================
        df_original = st.session_state.dados_original

        #col_data = None
        col_data = st.session_state.get("coluna_data")

        for coluna in df_original.columns:

            #if df_original[coluna].dtype == 'object':
            if (
                pd.api.types.is_object_dtype(df_original[coluna])
                or pd.api.types.is_datetime64_any_dtype(df_original[coluna])
            ):
                # tentativa com formato conhecido
                converted = pd.to_datetime(
                    df_original[coluna],
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )

                taxa_validos = converted.notna().sum() / len(df_original)

                # fallback controlado (sem warning)
                if taxa_validos < 0.5:
                    continue

                if taxa_validos > 0.9:
                    col_data = coluna
                    df_original[coluna] = converted
                    break
        #st.session_state["coluna_data"] = col_data
        if col_data is not None:
            st.session_state["coluna_data"] = col_data
        atributos_remover = st.session_state.get("atributos_removidos", [])

        # garante que tem os resumos (se não houver por algum motivo, gera agora)
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
        # PRIMEIRO LOOP → apenas interfaces de ordenação
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

                    # Recupera a ordem salva anteriormente, se existir
                    ordem_salva = st.session_state["ordem_valores"].get(col, lista_valores)

                    # Garante que os itens que sumiram (novos valores, etc.) ainda apareçam
                    for item in lista_valores:
                        if item not in ordem_salva:
                            ordem_salva.append(item)

                    # Mostra o componente já com a ordem salva
                    ordem_escolhida = sort_items(
                        items=ordem_salva,
                        direction="vertical",
                        key=f"sort_{col}"
                    )

                    # Se o usuário não interagiu (ordem_escolhida vazia), mantém a salva
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
        # SEGUNDO LOOP → gera os gráficos com base nas ordens
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
                orientation="h"  # barra horizontal
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
            fig.update_yaxes(tickfont=dict(size=13), automargin=True)
            fig.update_xaxes(showgrid=False, tickfont=dict(size=13))

            # ===== Exibição =====
            with cols[col_index]:
                st.markdown(
                    f"<p style='text-align: center; font-weight: bold;'>{col}</p>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
                
                # ===============================
                # Evolução temporal do atributo
                # ===============================
                col_data = st.session_state.get("coluna_data")
                

                if (
                    "coluna_data" in st.session_state
                    and st.session_state["coluna_data"] in df_original.columns
                ):
                    #st.session_state["coluna_data"] = col_data
                    top_valores = value_counts.index.tolist()

                    evolucao = analysis.gerar_evolucao_temporal(
                        df_original,
                        col_data,
                        col,
                        top_valores,
                        freq="Q"   # trimestral
                    )
                    #st.write("DEBUG evolução:", col, "→ linhas:", len(evolucao))
                    if not evolucao.empty:
                        data_inicio = evolucao[col_data].min()
                        data_fim = evolucao[col_data].max()

                        # traduz agregação
                        map_freq = {
                            "D": "Daily",
                            "W": "Weekly",
                            "M": "Monthly",
                            "Q": "Trimestres",
                            "Y": "Yearly"
                        }

                        freq_label = map_freq.get("Q", "Temporal")  # use sua variável freq aqui

                        periodo_texto = (
                            f"{freq_label} de "
                            f"{analysis.formatar_data_pt(data_inicio)} "
                            f"até {analysis.formatar_data_pt(data_fim)}"
                        )
                        fig_time = px.bar(
                            evolucao,
                            x=col_data,
                            y="count",
                            color=col,
                            height=220,
                            custom_data=["periodo_label"]
                        )
                        
                        fig_time.update_layout(
                            title=dict(
                                text=(
                                    f"{col} ao longo do tempo"
                                    f"<br><sup style='color:#6e7781; font-size:12px; font-weight:normal;'>"
                                    f"{periodo_texto}</sup>"
                                ),
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16)
                            ),
                            margin=dict(l=40, r=20, t=90, b=10),
                            xaxis_title="",
                            yaxis_title="",
                            legend_title="",
                            legend=dict(
                                font=dict(size=12)
                            ),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            barmode="group",
                            xaxis=dict(
                                rangeslider=dict( #mini-gráfico inferior
                                    visible=True,
                                    thickness=0.12
                                ),
                                type="date",
                                dtick="M24",       # ← intervalo de 2 anos
                                tickformat="%Y",   # mostra só o ano
                                ticklabelmode="period",
                                
                            )
                        )
                        fig_time.update_xaxes(
                            tickfont=dict(size=12)
                        )

                        fig_time.update_yaxes(
                            tickfont=dict(size=12)
                        )
                        fig_time.update_traces(
                            hovertemplate=(
                                "Período: %{customdata[0]}<br>"
                                "Valor: %{fullData.name}<br>"
                                "Quantidade: %{y}<extra></extra>"
                            )
                        )

                        st.plotly_chart(
                            fig_time,
                            use_container_width=True,
                            config={"displayModeBar": False}
                        )
            

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
    
    # Inicializa histórico de análises gerais
    if "analises_gerais" not in st.session_state:
        st.session_state.analises_gerais = []
    if "salvou_analise_geral" not in st.session_state:
        st.session_state.salvou_analise_geral = True
    
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
            
            st.session_state.salvou_analise_geral = False  # salvar só depois, na parte dos gráficos
            st.session_state.mostrar_regras = False
            
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
            st.info("Filtrando regras conforme meta-regras...")
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

                #lista para guardar os gráficos
                lista_graficos = []

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

                        #estrutura para salvar os gráficos desta "regra"
                        regra_plotada = {
                            "titulo": f"{atributo_antecedente} → {cons_val}",
                            "figs": {}
                        }

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

                                #guarda a figura
                                regra_plotada["figs"][medida] = fig
                        #depois do loop das 3 medidas, guarda essa regra na lista geral
                        lista_graficos.append(regra_plotada)

                #NOVO: salvar a análise geral no histórico
                if not st.session_state.get("salvou_analise_geral", True):
                    nome_arquivo_atual = st.session_state.get("arquivo_carregado", "(arquivo não registrado)")
                    nome_analise = f"Análise geral {len(st.session_state.analises_gerais) + 1}"

                    analise_geral = {
                        "nome": nome_analise,
                        "arquivo": nome_arquivo_atual,
                        "descricao": "Análise geral das meta-regras selecionadas.",
                        "parametros": {
                            "suporte_min": st.session_state.min_support,
                            "confianca_min": st.session_state.min_confidence,
                            "meta_regras": copy.deepcopy(st.session_state.regras),
                        },
                        "graficos": lista_graficos,  #os figs que preenchemos lá em cima
                    }

                    st.session_state.analises_gerais.append(analise_geral)
                    st.session_state.salvou_analise_geral = True

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
            # NOVO: detectar se já é datetime
            if pd.api.types.is_datetime64_any_dtype(df_original[c]):
                col_data = c
                break

            if df_original[c].dtype == 'object':
                serie = df_original[c].astype(str).str.strip()
                converted = pd.to_datetime(
                    serie,
                    errors='coerce'
                )

                taxa_validos = converted.notna().sum() / len(df_original)

                if taxa_validos > 0.7:
                    col_data = c
                    df_original[c] = converted
                    break

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
                    key="tipo_particionamento",
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
                        elif col_data is None:
                            st.error("Não foi possível detectar uma coluna de data para particionar.")
                        else:
                            # aqui uso o df_original e col_data já detectados lá em cima
                            marcos = [pd.to_datetime(m) for m in sorted(st.session_state.marcos_temporais)]
                            data_min = df_original[col_data].min()
                            data_max = df_original[col_data].max()
                            limites = [data_min] + marcos + [data_max]

                            particoes = []
                            for i in range(len(limites) - 1):
                                inicio = limites[i]
                                limite_superior  = limites[i + 1]
                                if i < len(limites) - 2:
                                    # partições normais: [inicio, fim)
                                    part = df_original[
                                        (df_original[col_data] >= inicio) &
                                        (df_original[col_data] < limite_superior)
                                    ].copy()
                                    fim_real = part[col_data].max() if not part.empty else inicio

                                else:
                                    # última partição: [inicio, fim]
                                    part = df_original[
                                        (df_original[col_data] >= inicio) &
                                        (df_original[col_data] <= limite_superior)
                                    ].copy()
                                    fim_real = limite_superior
                                particoes.append({"inicio": inicio, "fim": fim_real, "dados": part})
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

                    # CSS: deixa o number_input mais compacto, MAS preservando os botões +/-
                    st.markdown("""
                        <style>
                        /* A caixa externa continua grande o suficiente para os botões */
                        div[data-testid="stNumberInput"] > div {
                            max-width: 110px !important;    /* largura interna do input */
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # Organização em linha: número | botão | mensagem
                    col_num, col_btn, col_msg = st.columns([2, 1, 2])

                    with col_num:
                        # Subdivide a coluna: rótulo | input
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
                            st.success(f"{st.session_state.particoes_last_msg}")


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
                    # Ajusta fim apenas para exibição (exceto última partição)
                    #if i < len(st.session_state.particoes_temporais) - 1:
                    #    data_fim_exibicao = data_fim - pd.Timedelta(days=1)
                    #else:
                    data_fim_exibicao = data_fim

                    delta = relativedelta(data_fim_exibicao, data_ini)
                    partes = []
                    if delta.years > 0:
                        partes.append(f"{delta.years} ano{'s' if delta.years > 1 else ''}")
                    if delta.months > 0:
                        partes.append(f"{delta.months} mes{'es' if delta.months > 1 else ''}")
                    if delta.days > 0:
                        partes.append(f"{delta.days} dia{'s' if delta.days > 1 else ''}")
                    
                    duracao_calculada = ", ".join(partes)
                    duracao_texto = f" ({duracao_calculada if duracao_calculada else '0 dias'})"
                    
                    st.write(
                        f"Intervalo {i+1}: **{data_ini.strftime('%d/%m/%Y')}** → **{data_fim_exibicao.strftime('%d/%m/%Y')}** — "
                        f"{len(p['dados'])} registros {duracao_texto}"
                    )
                fig = analysis.gerar_barra_temporal(
                    st.session_state.particoes_temporais
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="barra_temporal"
                )    

                st.markdown("---")
                
                # --- BLOCO SEPARADO: Mostrar as opções de sup e conf para escolha ---
                # Cria 3 colunas: esquerda, central e direita
                st.subheader("Configuração do algoritmo para análise temporal")
                col_esq, col_central, col_dir = st.columns([2, 1, 1])

                with col_esq:
                    # Inputs lado a lado dentro da coluna central
                    col_s, col_c = st.columns([1,1])
                with col_s:
                    min_support_pct_temporal = numeric_text_input(
                        "Suporte mínimo (%)",
                        key="min_support_input",
                        value=st.session_state.min_support_temporal * 100,
                        min_value=0.0,
                        max_value=100.0
                    )
                    min_support_temporal = min_support_pct_temporal / 100.0
                with col_c:
                    min_confidence_pct_temporal = numeric_text_input(
                        "Confiança mínima (%)",
                        key="min_confidence_input",
                        value=st.session_state.min_confidence_temporal * 100,
                        min_value=0.0,
                        max_value=100.0
                    )
                    min_confidence_temporal = min_confidence_pct_temporal / 100.0
                st.session_state.min_support_temporal = min_support_temporal
                st.session_state.min_confidence_temporal = min_confidence_temporal
                #colunas = list(st.session_state.dados_processados.columns)
                
                if st.button("Gerar Análise Temporal", key="botao_analise_temporal"):
                    # Garante que a análise só será iniciada manualmente
                    st.session_state.analise_temporal_em_andamento = True
                    st.session_state.analise_temporal_pronta = False
                    st.session_state.analise_atual = None
            
            # --- BLOCO SEPARADO: EXECUTA ANÁLISE TEMPORAL QUANDO A FLAG ESTIVER ATIVA ---
            if st.session_state.get("analise_temporal_em_andamento", False) and not st.session_state.get("analise_temporal_pronta", False):

                atributos_regras = set()

                df_regras_base = st.session_state.get("base_regra")

                if df_regras_base is None or df_regras_base.empty:
                    st.info("Aguardando regras serem geradas...")
                else:

                    for ant in df_regras_base["antecedente"].unique():
                        if "=" in ant:
                            atributos_regras.add(ant.split("=", 1)[0])

                    for cons in df_regras_base["consequente"].unique():
                        if "=" in cons:
                            atributos_regras.add(cons.split("=", 1)[0])

                atributos_regras = list(atributos_regras)

                                

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


                def gerar_grafico_atributo(atributo, valor, particoes, df_geral):
                    
                    if atributo in df_geral.columns:
                        serie_geral = df_geral[atributo]
                        mask_geral = serie_geral.notna() & (serie_geral.astype(str) == valor)

                        suporte_geral = mask_geral.sum() / len(serie_geral) if len(serie_geral) > 0 else 0
                    else:
                        suporte_geral = 0

                    resultados_attr = []

                    for part in particoes:

                        df_part = part["dados"]

                        if df_part.empty or atributo not in df_part.columns:
                            continue

                        serie = df_part[atributo]

                        mask = serie.notna() & (serie.astype(str) == valor)

                        quantidade = mask.sum()
                        total = len(serie)

                        proporcao = quantidade / total if total > 0 else 0

                        if "inicio" in part:
                            periodo = (
                                f"{part['inicio'].strftime('%d/%m/%y')} - "
                                f"{part['fim'].strftime('%d/%m/%y')}"
                            )
                        else:
                            periodo = (
                                f"{part['data_min'].strftime('%d/%m/%y')} - "
                                f"{part['data_max'].strftime('%d/%m/%y')}"
                            )

                        resultados_attr.append({
                            "Periodo": periodo,
                            "proporcao": proporcao,
                            "quantidade": quantidade,
                            atributo: f"{atributo}={valor}"
                        })

                    if not resultados_attr:
                        return None

                    df_plot = pd.DataFrame(resultados_attr)
                    customdata_hover = df_plot[
                        ["Periodo", "proporcao", "quantidade"]
                    ]
                    fig = px.bar(
                        df_plot,
                        x="Periodo",
                        y="proporcao",
                        #text=df_plot["proporcao"].map(lambda v: f"{v:.2f}".replace(".", ",")),
                        text=df_plot["proporcao"].map(lambda v: f"{v:.2f}"),
                        color_discrete_sequence=["skyblue"],
                    )
                                     
                    fig.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        xref="paper",   
                        y0=suporte_geral,
                        y1=suporte_geral,
                        line=dict(color="rgba(255,0,0,0.6)", width=1.2, dash="dot")
                    )
                    fig.add_annotation(
                        x=0.95,                     
                        xref="paper",            
                        y=suporte_geral,
                        #text=f"{suporte_geral:.2f}".replace(".", ","),
                        text=f"{suporte_geral:.2f}",
                        showarrow=False,
                        font=dict(color="red", size=12),
                        xanchor="left",
                        yanchor="bottom",
                        xshift=2
                    )
                    fig.update_traces(
                        textposition="outside",
                        customdata=customdata_hover,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Suporte do atributo: <b>%{customdata[1]:.2f}</b><br>"
                            "Quantidade: <b>%{customdata[2]}</b>"
                            "<extra></extra>"
                        )
                    )

                    # gráfico pequeno 
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=80, b=30),
                        showlegend=False,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis_tickangle=45,
                        title=dict(
                            text=f"{atributo}={valor}",
                            x=0.5,
                            xanchor="center",
                            font=dict(
                                size=12,
                                color="#333",
                                family="Arial",
                                weight="normal"   # ← remove negrito
                            )
                        ),
                        yaxis=dict(
                            title="Suporte",
                            range=[0, 1.2],
                            showgrid=False
                        ),
                        xaxis_title=None,
                        uniformtext_minsize=8,
                        uniformtext_mode="hide",
                    )
                    #fig.update_yaxes(automargin=True)

                    return fig
                def gerar_grafico_antecedente_composto(antecedente, particoes, df_geral):

                    # ==========================================================
                    # Separa os itens do antecedente
                    # Exemplo:
                    # "pr_size=pr_l,first_pull=False"
                    # ->
                    # ["pr_size=pr_l", "first_pull=False"]
                    # ==========================================================

                    itens = [item.strip() for item in antecedente.split(",") if item.strip()]

                    condicoes = []

                    for item in itens:
                        partes = item.split("=", 1)

                        if len(partes) != 2:
                            continue

                        atributo, valor = partes
                        condicoes.append((atributo.strip(), valor.strip()))

                    if not condicoes:
                        return None

                    # ==========================================================
                    # SUPORTE GERAL DO ANTECEDENTE COMPOSTO
                    # ==========================================================

                    if df_geral.empty:
                        suporte_geral = 0
                    else:
                        mask_geral = pd.Series(True, index=df_geral.index)

                        for atributo, valor in condicoes:

                            if atributo not in df_geral.columns:
                                mask_geral &= False
                                continue

                            serie = df_geral[atributo]

                            mask_geral &= (
                                serie.notna() &
                                (serie.astype(str) == valor)
                            )

                        suporte_geral = (
                            mask_geral.sum() / len(df_geral)
                            if len(df_geral) > 0 else 0
                        )

                    # ==========================================================
                    # SUPORTE TEMPORAL
                    # ==========================================================

                    resultados_attr = []

                    for part in particoes:

                        df_part = part["dados"]

                        if df_part.empty:
                            continue

                        mask = pd.Series(True, index=df_part.index)

                        for atributo, valor in condicoes:

                            if atributo not in df_part.columns:
                                mask &= False
                                continue

                            serie = df_part[atributo]

                            mask &= (
                                serie.notna() &
                                (serie.astype(str) == valor)
                            )

                        quantidade = mask.sum()
                        total = len(df_part)

                        proporcao = (
                            quantidade / total
                            if total > 0 else 0
                        )

                        if "inicio" in part:
                            periodo = (
                                f"{part['inicio'].strftime('%d/%m/%y')} - "
                                f"{part['fim'].strftime('%d/%m/%y')}"
                            )
                        else:
                            periodo = (
                                f"{part['data_min'].strftime('%d/%m/%y')} - "
                                f"{part['data_max'].strftime('%d/%m/%y')}"
                            )

                        resultados_attr.append({
                            "Periodo": periodo,
                            "proporcao": proporcao,
                            "quantidade": quantidade,
                            "antecedente": antecedente
                        })

                    if not resultados_attr:
                        return None

                    # ==========================================================
                    # DATAFRAME DO GRÁFICO
                    # ==========================================================

                    df_plot = pd.DataFrame(resultados_attr)

                    customdata_hover = df_plot[
                        ["Periodo", "proporcao", "quantidade"]
                    ]

                    fig = px.bar(
                        df_plot,
                        x="Periodo",
                        y="proporcao",
                        text=df_plot["proporcao"].map(
                            lambda v: f"{v:.2f}"
                        ),
                        color_discrete_sequence=["skyblue"],
                    )

                    # ==========================================================
                    # LINHA DO SUPORTE GERAL
                    # ==========================================================

                    fig.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        xref="paper",
                        y0=suporte_geral,
                        y1=suporte_geral,
                        line=dict(
                            color="rgba(255,0,0,0.6)",
                            width=1.2,
                            dash="dot"
                        )
                    )

                    fig.add_annotation(
                        x=0.95,
                        xref="paper",
                        y=suporte_geral,
                        text=f"{suporte_geral:.2f}",
                        showarrow=False,
                        font=dict(
                            color="red",
                            size=12
                        ),
                        xanchor="left",
                        yanchor="bottom",
                        xshift=2
                    )

                    # ==========================================================
                    # HOVER
                    # ==========================================================

                    fig.update_traces(
                        textposition="outside",
                        customdata=customdata_hover,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Suporte do antecedente: "
                            "<b>%{customdata[1]:.2f}</b><br>"
                            "Quantidade: <b>%{customdata[2]}</b>"
                            "<extra></extra>"
                        )
                    )

                    # ==========================================================
                    # LAYOUT
                    # ==========================================================

                    fig.update_layout(
                        height=300,
                        margin=dict(
                            l=10,
                            r=10,
                            t=80,
                            b=30
                        ),
                        showlegend=False,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis_tickangle=45,
                        title=dict(
                            text=antecedente,
                            x=0.5,
                            xanchor="center",
                            font=dict(
                                size=12,
                                color="#333",
                                family="Arial",
                                weight="normal"
                            )
                        ),
                        yaxis=dict(
                            title="Suporte",
                            range=[0, 1.2],
                            showgrid=False
                        ),
                        xaxis_title=None,
                        uniformtext_minsize=8,
                        uniformtext_mode="hide",
                    )

                    return fig
                #if "regras_temporais_mineradas" not in st.session_state:
                if (
                    "regras_temporais_mineradas" not in st.session_state
                    or len(st.session_state.regras_temporais_mineradas)
                    != len(st.session_state.particoes_temporais)
                ):

                    st.session_state.regras_temporais_mineradas = []

                    for part in st.session_state.particoes_temporais:

                        df_part = part["dados"].copy()

                        if df_part.empty:
                            st.session_state.regras_temporais_mineradas.append(pd.DataFrame())
                            continue

                        if col_data in df_part.columns:
                            df_part = df_part.drop(columns=[col_data])

                        df_part_tratado, _, _ = analysis.preparar_dados_para_mineracao_from_df(df_part)

                        df_regras_part = analysis.gerar_regras_com_mlxtend2(
                            df_part_tratado,
                            st.session_state.min_support_temporal,
                            st.session_state.min_confidence_temporal
                        )
                        if not df_regras_part.empty:

                            df_regras_part["id_regra"] = (
                                df_regras_part["antecedente"]
                                + "###"
                                + df_regras_part["consequente"]
                            )

                            df_regras_part = df_regras_part.set_index("id_regra")
                        #st.session_state.regras_temporais_mineradas.append(df_regras_part)
                        st.session_state.regras_temporais_mineradas.append({
                            "dados_tratados": df_part_tratado,
                            "regras": df_regras_part
                        })

                # === Análise Temporal das Regras ===
                #st.write("ENTREI NA ANALISE TEMPORAL")
                for idx_regra, regra_user in enumerate(st.session_state.regras):
                    #st.write("Meta-regra", idx_regra)
                    ant_attr = regra_user["antecedente"]
                    cons_attr = regra_user["consequente"]

                    #base_geral_filtrada = st.session_state.base_regra[
                    #    st.session_state.base_regra["antecedente"].str.match(f"^{ant_attr}=.+$")
                    #]
                    base_geral_filtrada = st.session_state.base_regra[
                        (st.session_state.base_regra["antecedente"].str.match(f"^{ant_attr}=.+$"))
                        &
                        (st.session_state.base_regra["consequente"].str.match(f"^{cons_attr}=.+$"))
                    ]

                    if base_geral_filtrada.empty:
                        st.warning(f"Nenhuma regra encontrada na base geral para {ant_attr} → {cons_attr}")
                        continue
                    def safe_key(text):
                        return (
                            str(text)
                            .replace(" ", "_")
                            .replace("=", "_")
                            .replace(">", "")
                            .replace("<", "")
                        )
                    for ant_val, grupo_ant in base_geral_filtrada.groupby("antecedente"):
                        for cons_val, grupo_cons_geral in grupo_ant.groupby("consequente"):
                            st.markdown(
                                f"<h5 style='text-align:center; color:#222; margin-top:10px; margin-bottom:4px;'>"
                                f"Regra: {ant_val} → {cons_val}</h5>",
                                unsafe_allow_html=True
                            )
                            combo_id = f"{idx_regra}_{ant_val}_{cons_val}"

                            medidas_particoes = []

                            for i, part in enumerate(st.session_state.particoes_temporais):

                                info_part = st.session_state.regras_temporais_mineradas[i]

                                df_regras_part = info_part["regras"]

                                df_part_tratado = info_part["dados_tratados"]

                                df_part = part["dados"].copy()
                                qtd_registros = len(df_part)
                                attr_cons, val_cons = cons_val.split("=")

                                if attr_cons in df_part_tratado.columns:
                                    suporte_consequente = (
                                        df_part_tratado[attr_cons] == val_cons
                                    ).mean()
                                else:
                                    suporte_consequente = 0

                                if df_regras_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0,
                                    "sup_consequente": suporte_consequente})
                                    continue

                                df_filtrado_part = df_regras_part[
                                    (df_regras_part["antecedente"] == ant_val) &
                                    (df_regras_part["consequente"] == cons_val)
                                ]

                                if df_filtrado_part.empty:
                                    medidas_particoes.append({"suporte":0, "confianca":0, "lift":0,  "sup_consequente": suporte_consequente,"qtd_registros": qtd_registros})
                                else:
                                    row = df_filtrado_part.iloc[0]
                                    medidas_particoes.append({
                                        "suporte": row["suporte"],
                                        "confianca": row["confianca"],
                                        "lift": row["lift"],
                                        #alter aqui
                                        "sup_consequente": suporte_consequente,
                                        "qtd_registros": qtd_registros
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
                                periodos.append(f"{ini.strftime('%d/%m/%y')} - {fim.strftime('%d/%m/%y')}")
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
                                        #text=df_medidas[medida].apply(lambda x: f"{x:.2f}".replace(".", ",")),
                                        text=df_medidas[medida].apply(lambda x: f"{x:.2f}"),
                                        title=medida.capitalize(),
                                    )

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
                                        x=x_final,                # depois da última barra
                                        y=y_ref,                  # mesma altura da linha
                                        #text=f"{round2(y_ref)}",
                                        #text=f"{y_ref:.2f}".replace(".", ","),      # valor formatado
                                        text=f"{y_ref:.2f}",
                                        showarrow=False,
                                        font=dict(color="red", size=12),
                                        xanchor="left",           # texto “depois” da linha
                                        yanchor="bottom",
                                        xshift=10                 # desloca levemente para a direita
                                    )

                                    fig.update_traces(
                                        selector=dict(type="bar"),
                                        marker_color=cores_fixas[medida],
                                        texttemplate="%{text}",
                                        textposition="outside",
                                        textfont=dict(size=12),
                                        cliponaxis=False,
                                        hovertemplate=(
                                            " <b>%{customdata[0]}</b><br>"
                                            "Suporte: <b>%{customdata[1]:.2f}</b><br>"
                                            "Confiança: <b>%{customdata[2]:.2f}</b><br>"
                                            "Lift: <b>%{customdata[3]:.2f}</b><br>"
                                            "Sup. Consequente: <b>%{customdata[4]:.2f}</b><br>"
                                            "Quantidade de Registros: <b>%{customdata[5]}</b>"
                                            "<extra></extra>"
                                        ),
                                        customdata=df_medidas[
                                            ["Periodo", 
                                            "suporte", 
                                            "confianca", 
                                            "lift", 
                                            "sup_consequente", 
                                            "qtd_registros"]
                                        ]
                                    )                                

                                    # Layout e formatação geral
                                    fig.update_layout(
                                        height=350,
                                        margin=dict(l=10, r=10, t=50, b=50),
                                        title=dict(
                                            text=medida.capitalize(),
                                            x=0.5,  # Centraliza título
                                            xanchor="center",
                                            font=dict(size=14, color="#333", family="Arial", weight="normal")
                                        ),
                                        legend=dict(                 
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="left",
                                            x=0.5
                                        ),
                                         xaxis=dict(
                                            tickangle=45,
                                            title=None,
                                            tickfont=dict(size=12),
                                            showgrid=False   #  remove grade vertical
                                            #zeroline=False    #  remove linha zero
                                        ),
                                        yaxis=dict(
                                            title=None,
                                            tickfont=dict(size=12),
                                            showgrid=False   #  remove grade horizontal
                                            #zeroline=False    #  remove linha zero
                                        ),
                                        plot_bgcolor="white",
                                        paper_bgcolor="white",
                                        showlegend=True
                                    )

                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"{idx_regra}_{ant_val}_{cons_val}_{medida}_{i}"
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
                            # ======================================
                            # GRÁFICOS DOS ATRIBUTOS DA REGRA
                            # ======================================

                            st.markdown(
                                """
                                <h6 style='text-align:center; margin-top:15px; margin-bottom:5px; color:#444;'>
                                    Suporte Temporal do Antecedente e Consequente
                                </h6>
                                """,
                                unsafe_allow_html=True
                            )
                            col_attr1, col_attr2, col_attr3 = st.columns(3)
                            
                            # separa atributo e valor da regra
                            attr_ant, val_ant = ant_val.split("=")
                            
                            attr_cons, val_cons = cons_val.split("=")

                            # ==========================================================
                            # GRÁFICO DO ANTECEDENTE
                            # ==========================================================

                            with col_attr1:

                                # Verifica se o antecedente possui mais de um item
                                itens_antecedente = [
                                    item.strip()
                                    for item in ant_val.split(",")
                                    if item.strip()
                                ]

                                if len(itens_antecedente) > 1:

                                    # Antecedente composto
                                    fig_attr = gerar_grafico_antecedente_composto(
                                        ant_val,
                                        st.session_state.particoes_temporais,
                                        df_original
                                    )

                                else:

                                    # Antecedente simples
                                    attr_ant, val_ant = ant_val.split("=", 1)

                                    fig_attr = gerar_grafico_atributo(
                                        attr_ant,
                                        val_ant,
                                        st.session_state.particoes_temporais,
                                        df_original
                                    )

                                if fig_attr:
                                    st.plotly_chart(
                                        fig_attr,
                                        use_container_width=True,
                                        key=f"attr_ant_{safe_key(combo_id)}"
                                    )


                            # ==========================================================
                            # GRÁFICO DO CONSEQUENTE
                            # ==========================================================

                            with col_attr3:

                                attr_cons, val_cons = cons_val.split("=", 1)

                                fig_cons = gerar_grafico_atributo(
                                    attr_cons,
                                    val_cons,
                                    st.session_state.particoes_temporais,
                                    df_original
                                )

                                if fig_cons:
                                    st.plotly_chart(
                                        fig_cons,
                                        use_container_width=True,
                                        key=f"attr_cons_{safe_key(combo_id)}"
                                    )
                nome_arquivo_atual = st.session_state.get("arquivo_carregado", "(arquivo não registrado)")
                nome_analise = f"Análise {len(st.session_state.analises_temporais) + 1}"
                analise_atual["nome"] = nome_analise
                analise_atual["arquivo"] = nome_arquivo_atual
                analise_atual["parametros"] = {
                    "suporte_min_temporal": st.session_state.min_support_temporal,
                    "confianca_min_temporal": st.session_state.min_confidence_temporal,
                    "meta_regras": copy.deepcopy(st.session_state.regras),
                    "particoes": copy.deepcopy(st.session_state.particoes_temporais),
                    "tipo_particionamento": st.session_state.get("tipo_particionamento", "Não informado")
                }

                st.session_state.analises_temporais.append(copy.deepcopy(analise_atual))
                # 🔹 Marca como concluída
                st.session_state.analise_temporal_em_andamento = False
                st.session_state.analise_temporal_pronta = True
                st.success(f"{nome_analise} salva com sucesso!")
# ---------- Aba 4: Histórico de Análises Gerais ----------
# --- ABA: Histórico de Análises Gerais ---
elif tab == "Histórico de Análises Gerais":
    st.subheader("Histórico de Análises Gerais")

    if "analises_gerais" not in st.session_state or not st.session_state.analises_gerais:
        st.info("Nenhuma análise geral foi realizada ainda.")
    else:
        # Cria uma aba para cada análise salva
        abas_hist = st.tabs([a["nome"] for a in st.session_state.analises_gerais])

        for i, analise in enumerate(st.session_state.analises_gerais):
            with abas_hist[i]:
                st.markdown(f"### {analise['nome']}")

                # Arquivo analisado
                st.markdown(
                    f"**Arquivo analisado:** {analise.get('arquivo', '(não informado)')}"
                )

                # Descrição (se houver)
                if analise.get("descricao"):
                    st.write(analise["descricao"])

                st.markdown("---")

                # Parâmetros principais
                params = analise.get("parametros", {})
                suporte_min = params.get("suporte_min", None)
                confianca_min = params.get("confianca_min", None)
                tempos = params.get("tempos", {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    if suporte_min is not None:
                        st.markdown(f"**Suporte mínimo:** {suporte_min:.2%}")
                    else:
                        st.markdown("**Suporte mínimo:** -")
                with col2:
                    if confianca_min is not None:
                        st.markdown(f"**Confiança mínima:** {confianca_min:.2%}")
                    else:
                        st.markdown("**Confiança mínima:** -")
                with col3:
                    tempo_total = tempos.get("tempo_total", None)
                    if tempo_total is not None:
                        st.markdown(f"**Tempo total:** {tempo_total:.2f} s")
                    else:
                        st.markdown("**Tempo total:** -")

                st.markdown("---")

                # Meta-regras usadas na análise
                st.markdown("**Meta-regras analisadas:**")
                meta_regras = params.get("meta_regras", [])
                if meta_regras:
                    for regra in meta_regras:
                        st.write(f"• {regra['antecedente']} → {regra['consequente']}")
                else:
                    st.write("_Nenhuma meta-regra registrada nesta análise._")

                # Tempos detalhados (se quiser mostrar)
                if tempos:
                    st.markdown("---")
                    st.markdown("**Tempos de execução:**")
                    if "tempo_regras" in tempos:
                        st.write(f"- Geração de regras: {tempos['tempo_regras']:.2f} s")
                    if "tempo_filtro" in tempos:
                        st.write(f"- Filtragem: {tempos['tempo_filtro']:.2f} s")
                    if "tempo_graficos" in tempos:
                        st.write(f"- Geração dos gráficos: {tempos['tempo_graficos']:.2f} s")

                st.markdown("---")

                # Reexibir gráficos salvos
                st.markdown("### Gráficos da Análise")

                for idx_regra, regra_plotada in enumerate(analise.get("graficos", [])):
                    st.markdown(f"#### {regra_plotada.get('titulo', 'Regra')}")

                    cols = st.columns(3)
                    for j, medida in enumerate(["suporte", "confianca", "lift"]):
                        fig = regra_plotada["figs"].get(medida)
                        if fig is not None:
                            with cols[j]:
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"hist_geral_{i}_{idx_regra}_{medida}"
                                )
# ---------- Aba 5: Histórico de Análises Temporais ----------
elif tab == "Histórico de Análises Temporais":
    st.subheader("Histórico de Análises Temporais")

    if "analises_temporais" not in st.session_state or not st.session_state.analises_temporais:
        st.info("Nenhuma análise temporal foi realizada ainda.")
    else:
        abas_hist = st.tabs([a["nome"] for a in st.session_state.analises_temporais])

        for i, analise in enumerate(st.session_state.analises_temporais):
            with abas_hist[i]:
                st.markdown(f"### {analise['nome']}")

                
                col1, col2, col3 = st.columns(3)
                with col1:
                    #mostrar arquivo analisado
                    st.markdown(
                        f"**Arquivo analisado:** {analise.get('arquivo', '(não informado)')}"
                    )

                    st.write(analise["descricao"])
                    st.markdown(f"**Suporte mínimo:** {analise['parametros']['suporte_min_temporal']}")
                    st.markdown(f"**Confiança mínima:** {analise['parametros']['confianca_min_temporal']}")
                    st.markdown(f"**Tipo de particionamento:** {analise['parametros'].get('tipo_particionamento', '-')}")
                with col2:
                    st.markdown("**Meta-regras analisadas:**")
                    for regra in analise['parametros']['meta_regras']:
                        st.write(f"• {regra['antecedente']} → {regra['consequente']}")
                with col3:
                    st.markdown("**Partições geradas:**")
                    
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