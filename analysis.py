# analysis.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import io
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import subprocess
import os
import re

def preparar_dados_para_mineracao_from_df(df_original):
    """
    Remove atributos numéricos (com muitas categorias) e colunas de data.
    Retorna: df_discretizado, dados_removidos (DataFrame), atributos_remover (list).
    """
    df = df_original.copy()

    atributos_numericos = []
    atributos_de_data = []

    for coluna in df.columns:
        # detectar numéricos com muitas categorias
        if pd.api.types.is_numeric_dtype(df[coluna]) and df[coluna].nunique() > 10:
            atributos_numericos.append(coluna)
        # detectar data (conversão com maioria dos valores válidos)
        else:
            try:
                converted = pd.to_datetime(df[coluna], errors='coerce', format=None)
                if converted.notna().sum() > len(df) * 0.9:
                    atributos_de_data.append(coluna)
            except Exception:
                pass

    atributos_remover = atributos_numericos + atributos_de_data
    dados_removidos = df[atributos_remover].copy()
    df_discretizado = df.drop(columns=atributos_remover)

    # Converte booleanos para str para evitar problemas no OHE
    for col in df_discretizado.select_dtypes(include=['bool']).columns:
        df_discretizado[col] = df_discretizado[col].astype(str)

    return df_discretizado, dados_removidos, atributos_remover


def preparar_para_apriori(df):
    """Transforma dataframe categórico em one-hot encoded para mlxtend"""
    df_oht = pd.get_dummies(df, prefix_sep='=')
    return df_oht

# --- trata antecedentes compostos ---
def parse_antecedentes(ant_str):
    """Divide uma string de antecedente composto (ex: 'first_pull, lifetime') em lista."""
    return [a.strip() for a in str(ant_str).split(",") if a.strip()]

# --- trata antecedente composto ("a, b, c") ---
def _parse_antecedentes(ant):
    # ant pode vir como "a, b" (string) ou lista (se já vindo do state)
    if isinstance(ant, list):
        return [s.strip() for s in ant if s and str(s).strip()]
    return [s.strip() for s in str(ant).split(",") if s and s.strip()]

def gerar_regras_com_r(df, sup=0.01, conf=0.01, script_path="gerar_regras.R",
                       lhs_attr=None, rhs_attr=None):
    """
    Gera regras chamando gerar_regras.R e passando sup/conf como argumentos.
    Usa arquivos temporários (robusto para Streamlit) e retorna no mesmo formato do mlxtend.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # salva input temporário em UTF-8
    tmp_in = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_in.close()
    tmp_out.close()

    try:
        df.to_csv(tmp_in.name, index=False, encoding="utf-8")

        # chama o R passando 4 argumentos: input, output, sup, conf
        cmd = [
            "Rscript",
            script_path,
            tmp_in.name,
            tmp_out.name,
            str(sup),
            str(conf),
            lhs_attr or "",
            rhs_attr or ""
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("=== ERRO R (stdout) ===")
            print(result.stdout)
            print("=== ERRO R (stderr) ===")
            print(result.stderr)
            raise RuntimeError("Erro ao executar Rscript")

        # lê resultado
        if os.path.exists(tmp_out.name) and os.path.getsize(tmp_out.name) > 0:
            df_regras = pd.read_csv(tmp_out.name, encoding="utf-8")
        else:
            df_regras = pd.DataFrame()

        # padroniza colunas para o formato do app
        if not df_regras.empty:
            # traduz nomes e garante ordem
            rename_map = {
                'support': 'suporte',
                'confidence': 'confianca'
            }
            df_regras.rename(columns=rename_map, inplace=True, errors='ignore')

            cols = ['antecedente', 'consequente', 'suporte', 'confianca', 'lift']
            df_regras = df_regras[[c for c in cols if c in df_regras.columns]]

        return df_regras

    finally:
        # cleanup
        for p in (tmp_in.name, tmp_out.name):
            try:
                os.unlink(p)
            except OSError:
                pass

def gerar_regras_com_mlxtend(df, sup, conf):
    """Gera regras e retorna DataFrame formatado com métricas traduzidas."""
    df_oht = preparar_para_apriori(df)
    if df_oht.shape[1] == 0:
        return pd.DataFrame()  # nada para minerar

    # Apriori e association rules
    frequent_itemsets = apriori(df_oht, min_support=float(sup), use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=float(conf))
    if regras.empty:
        return pd.DataFrame()

    df_regras = regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    df_regras['antecedente'] = df_regras['antecedents'].apply(
        lambda x: ','.join([reconstruir_attr_valor(i) for i in x])
    )
    df_regras['consequente'] = df_regras['consequents'].apply(
        lambda x: ','.join([reconstruir_attr_valor(i) for i in x])
    )
    df_regras.rename(columns={'support': 'suporte', 'confidence': 'confianca'}, inplace=True)

    # seleciona e retorna
    cols = ['antecedente', 'consequente', 'suporte', 'confianca', 'lift']
    return df_regras[cols]

def gerar_regras_com_mlxtend2(df, sup, conf):
    """Gera regras e retorna DataFrame formatado com métricas traduzidas."""
    # helper para reconstruir um item qualquer em "atributo=valor"
    def _item_para_atributo_valor(item, original_cols):
        s = str(item)
        # já está no formato "attr=val"
        if "=" in s:
            return s.strip()
        # se veio como "attr_val"
        if "_" in s:
            attr, val = s.split("_", 1)
            return f"{attr}={val}"
        # se o nome bate com uma coluna original (possível binarização de True)
        if s in original_cols:
            return f"{s}=True"
        # fallback: retornar "item=True" (pode ajustar se preferir outro comportamento)
        return f"{s}=True"

    df_oht = preparar_para_apriori(df)
    if df_oht.shape[1] == 0:
        return pd.DataFrame()  # nada para minerar

    # Apriori e association rules
    frequent_itemsets = apriori(df_oht, min_support=float(sup), use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=float(conf))
    if regras.empty:
        return pd.DataFrame()

    df_regras = regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()

    # lista de colunas originais (para decidir quando assumir "=True")
    original_cols = df.columns.tolist()

    # normaliza antecedente e consequente para "atributo=valor"
    df_regras['antecedente'] = df_regras['antecedents'].apply(
        lambda x: ','.join([_item_para_atributo_valor(i, original_cols) for i in sorted(list(x))])
    )
    df_regras['consequente'] = df_regras['consequents'].apply(
        lambda x: ','.join([_item_para_atributo_valor(i, original_cols) for i in sorted(list(x))])
    )

    df_regras.rename(columns={'support': 'suporte', 'confidence': 'confianca'}, inplace=True)

    # seleciona e retorna
    cols = ['antecedente', 'consequente', 'suporte', 'confianca', 'lift']
    return df_regras[cols]

def listar_itens_possiveis(df):
    """
    Retorna todos os itens possíveis (atributo=valor) a partir das colunas
    e valores únicos do DataFrame original.
    """
    itens = []
    for col in df.columns:
        valores_unicos = df[col].dropna().unique()
        for v in valores_unicos:
            itens.append(f"{col}={v}")
    return sorted(itens)

# Extrair atributos categóricos no formato "atributo=valor"
# Gera opções para a montagem do selectbox para a definição da regra que se quer analisar 
def gerar_opcoes(df):
    opcoes = []
    for col in df.columns:
        valores = df[col].dropna().unique()
        for v in valores:
            opcoes.append(f"{col}={v}")
    return opcoes


def exportar_regras_para_excel_bytes(df):
    """Exporta um DataFrame para bytes de Excel."""
    if df is None or df.empty:
        raise ValueError("O DataFrame está vazio. Não há regras para exportar.")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Regras")
    output.seek(0)
    return output.getvalue()

def filtrar_regras_por_item(df_regras, antecedentes=None, consequentes=None):
    """
    Filtra regras de acordo com itens (ex.: "merged=True") em antecedente e/ou consequente.
    - antecedentes: lista de itens para o antecedente. Se fornecida, o antecedente deve ser exatamente igual ao conjunto.
    - consequentes: lista de itens para o consequente.
        * Se len==1: consequente deve ser exatamente esse item.
        * Se len>=2: aceita qualquer subconjunto não-vazio do conjunto escolhido, sem extras.
    Retorna um DataFrame filtrado.
    """
    def to_set(s):
        if isinstance(s, set):
            return s
        if pd.isna(s) or s == "":
            return set()
        return set(map(str.strip, str(s).split(",")))

    ant_sel = set(antecedentes) if antecedentes else None
    cons_sel = set(consequentes) if consequentes else None

    def ok(row):
        ant = to_set(row["antecedente"])
        cons = to_set(row["consequente"])

        # Filtra antecedente: igualdade exata, se fornecido
        if ant_sel is not None:
            if ant != ant_sel:
                return False

        # Filtra consequente conforme as regras descritas
        if cons_sel is not None:
            if len(cons_sel) == 1:
                # precisa ser exatamente o único item escolhido
                if cons != cons_sel:
                    return False
            else:
                # aceita subconjuntos não-vazios, sem extras
                if not cons or not cons.issubset(cons_sel):
                    return False

        return True

    mask = df_regras.apply(ok, axis=1)
    return df_regras[mask].copy()

def filtrar_regras_por_atributo2(df_regras, regras_usuario):
    """
    Filtra regras do DataFrame de acordo com as regras completas selecionadas pelo usuário.
    Cada regra do usuário é do tipo: {'antecedente': 'atributo', 'consequente': 'atributo'}.

    - Seleciona apenas regras simples (sem vírgula) no antecedente e consequente.
    """
    import pandas as pd

    if not regras_usuario or df_regras.empty:
        return pd.DataFrame(columns=df_regras.columns)

    filtros = []
    for regra in regras_usuario:
        atr_ant = regra["antecedente"]
        atr_cons = regra["consequente"]

        # filtra apenas regras simples com o atributo escolhido
        filtro = df_regras[
            df_regras['antecedente'].str.match(f"^{atr_ant}=.+$") & (~df_regras['antecedente'].str.contains(",")) &
            df_regras['consequente'].str.match(f"^{atr_cons}=.+$") & (~df_regras['consequente'].str.contains(","))
        ]
        filtros.append(filtro)

    if filtros:
        df_filtrado = pd.concat(filtros, ignore_index=True).drop_duplicates()
    else:
        df_filtrado = pd.DataFrame(columns=df_regras.columns)

    return df_filtrado

def _extract_attr_set(side: str) -> set:
    """
    Extrai o conjunto de nomes de atributos de uma string do lado da regra,
    ex.: "first_pull=True,lifetime=long" -> {"first_pull","lifetime"}.
    Ignora espaços e respeita caixa-insensível.
    """
    if side is None or pd.isna(side):
        return set()
    # procura padrões tipo atributo=valor
    attrs = re.findall(r'([A-Za-z0-9_]+)\s*=', str(side))
    return set(a.strip().lower() for a in attrs)

def _parse_meta_antecedente(meta_ant: str) -> set:
    """
    Converte a meta-seleção do usuário em conjunto de atributos
    ex.: "first_pull, lifetime" -> {"first_pull","lifetime"}.
    """
    if meta_ant is None:
        return set()
    parts = [p.strip().lower() for p in str(meta_ant).split(",") if p and p.strip()]
    return set(parts)

def filtrar_regras_por_atributo(df_regras: pd.DataFrame, lista_meta_regras):
    """
    Filtra df_regras para conter SOMENTE regras que correspondem
    EXATAMENTE à composição de atributos escolhida nas meta-regras.
    - Antecedente: conjunto de atributos EXACTO.
    - Consequente: conjunto EXACTO com 1 atributo (o escolhido).
    Funciona para meta-regras simples e compostas.
    """
    if df_regras is None or df_regras.empty:
        return pd.DataFrame()

    df = df_regras.copy()

    # Extrai conjuntos de atributos de cada lado, uma vez só
    if "antecedente_attrs" not in df.columns or "consequente_attrs" not in df.columns:
        df["antecedente_attrs"] = df["antecedente"].apply(_extract_attr_set)
        df["consequente_attrs"] = df["consequente"].apply(_extract_attr_set)

    blocos = []
    for meta in lista_meta_regras:
        meta_ant = meta["antecedente"]
        meta_cons = meta["consequente"]

        ant_set = _parse_meta_antecedente(meta_ant)            # ex.: {"first_pull"} ou {"first_pull","lifetime"}
        cons_set = {str(meta_cons).strip().lower()}            # ex.: {"merged"}

        tam_esperado = len(ant_set)
        mask = (
            (df["antecedente_attrs"].apply(lambda s: s == ant_set or (len(ant_set) == 1 and s == ant_set)))
            & (df["consequente_attrs"].apply(lambda s: s == cons_set))
            & (df["antecedente_attrs"].apply(lambda s: len(s) == tam_esperado))
        )

        subset = df[mask].copy()
        if not subset.empty:
            subset["meta_regra"] = f"{meta_ant} → {meta_cons}"
            blocos.append(subset)

    if not blocos:
        return pd.DataFrame()

    out = pd.concat(blocos, ignore_index=True)

    # Remove duplicatas só com base em colunas textuais/numéricas (sem sets)
    dup_cols = [c for c in out.columns if out[c].map(lambda x: not isinstance(x, set)).all()]
    out = out.drop_duplicates(subset=dup_cols)

    # Opcional: remover colunas auxiliares antes de devolver
    return out.drop(columns=["antecedente_attrs", "consequente_attrs"], errors="ignore")



def reconstruir_attr_valor(col_name):
    """
    Converte o nome da coluna codificada de volta para 'atributo=valor'.
    Ex.: 'Status_Closed' -> 'Status=Closed'
    """
    if "_" in col_name:
        attr, val = col_name.split("_", 1)
        return f"{attr}={val}"
    return col_name


def plotar_comparacoes(resultados, regra_base):
    """
    Gera gráficos comparando suporte, confiança e lift da regra-base
    com as partições temporais.
    """
    # Valores da regra base (linha vermelha nos gráficos)
    suporte_base = regra_base["suporte"]
    confianca_base = regra_base["confianca"]
    lift_base = regra_base["lift"]

    # Valores por partição
    suporte_part = [p["regras"][0]["suporte"] for p in resultados if p["regras"]]
    confianca_part = [p["regras"][0]["confianca"] for p in resultados if p["regras"]]
    lift_part = [p["regras"][0]["lift"] for p in resultados if p["regras"]]

    labels = [p["particao"] for p in resultados if p["regras"]]

    # === Gráfico Suporte ===
    plt.figure(figsize=(8,5))
    plt.bar(labels, suporte_part, color="skyblue")
    plt.axhline(y=suporte_base, color="red", linestyle="--", label="Base Geral")
    plt.title(f"Variação do Suporte\n{regra_base['antecedente']} → {regra_base['consequente']}")
    plt.ylabel("Suporte")
    plt.legend()
    plt.show()

    # === Gráfico Confiança ===
    plt.figure(figsize=(8,5))
    plt.bar(labels, confianca_part, color="lightgreen")
    plt.axhline(y=confianca_base, color="red", linestyle="--", label="Base Geral")
    plt.title(f"Variação da Confiança\n{regra_base['antecedente']} → {regra_base['consequente']}")
    plt.ylabel("Confiança")
    plt.legend()
    plt.show()

    # === Gráfico Lift ===
    plt.figure(figsize=(8,5))
    plt.bar(labels, lift_part, color="orange")
    plt.axhline(y=lift_base, color="red", linestyle="--", label="Base Geral")
    plt.title(f"Variação do Lift\n{regra_base['antecedente']} → {regra_base['consequente']}")
    plt.ylabel("Lift")
    plt.legend()
    plt.show()


def gerar_analise_temporal(df, regra_selecionada, particoes):
    """
    Função principal que roda a análise temporal e gera gráficos automaticamente.
    """
    resultados = []  # <- aqui entram os resultados por partição que você já gera no seu código

    # >>> Lógica de mineração temporal que você já tem
    # resultados = minerar_regras_por_particao(df, particoes, regra_selecionada)

    # Depois que resultados estiver pronto, chamamos os gráficos:
    if resultados and regra_selecionada:
        plotar_comparacoes(resultados, regra_selecionada)
    else:
        print("⚠ Nenhum resultado encontrado para análise temporal.")

# === função robusta para reconstruir "atributo=valor" ===
def reconstruir_attr_valor(item, df_part=None):
    """
    Recebe item que pode ser:
      - string como "attr_val" ou "attr=val"
      - frozenset/set (ex.: frozenset({'attr_val'}) ou frozenset({'attr=val'}))
    Retorna sempre no formato 'atributo=valor'.
    Se não for possível inferir, retorna str(item).
    """
    import pandas as pd

    # trata NaN
    try:
        if pd.isna(item):
            return item
    except Exception:
        pass

    # --- Caso seja conjunto (mlxtend retorna frozenset/set)
    if isinstance(item, (set, frozenset, list, tuple)):
        itens = list(item)
        if len(itens) == 0:
            return ""
        if len(itens) == 1:
            return reconstruir_attr_valor(itens[0], df_part)
        return ",".join([reconstruir_attr_valor(x, df_part) for x in itens])

    s = str(item).strip()

    # --- já está no formato correto
    if "=" in s:
        return s

    # --- dummy encoding padrão: attr_val -> attr=val
    if "_" in s:
        attr, val = s.split("_", 1)
        return f"{attr}={val}"

    # --- às vezes vem com espaço: "attr val"
    if " " in s:
        parts = s.split()
        if len(parts) == 2:
            return f"{parts[0]}={parts[1]}"

    # --- caso especial: se for nome de coluna do DF da partição
    if df_part is not None and s in df_part.columns:
        col = df_part[s]

        # normaliza valores booleanos/numéricos
        uniques = pd.unique(col.dropna())
        uniq_set = set(str(u) for u in uniques)

        # caso booleano ou binário 0/1
        if uniq_set.issubset({"0", "1", "True", "False"}):
            return f"{s}=True"

        # caso categórico (string) com poucos valores
        if col.dtype == object or len(uniques) <= 10:
            most = col.dropna().astype(str).mode()
            if not most.empty:
                val = most.iloc[0]
                return f"{s}={val}"

    # fallback
    return s

# --- Função utilitária para normalizar itemset ---
def normalizar_regra(s, df_part=None):
    """
    Recebe string ou frozenset e devolve 'atributo=valor'
    usando o contexto da partição para resolver 'attr_val' -> 'attr=val'.
    """
    if pd.isna(s):
        return ""
    ss = str(s).strip()

    # Caso frozenset({'a_b', 'c_d'}) ou set(...)
    if ss.startswith("frozenset") or ss.startswith("set"):
        start = ss.find("{")
        end = ss.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = ss[start+1:end]
            partes = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
            return ",".join([reconstruir_attr_valor(p, df_part=df_part) for p in partes])

    # Caso string simples (ex: "attr_val" ou "attr=val")
    return reconstruir_attr_valor(ss, df_part=df_part)

# --- Geração das medidas por partição ---
def extrair_vals(df, coluna, attr):
    """Retorna lista ordenada de valores (strings) do atributo `attr`
    presente nas células da coluna `coluna`. Suporta itens compostos 'a=1,b=2'."""
    vals = set()
    if coluna not in df.columns:
        return []
    for s in df[coluna].dropna().astype(str):
        for item in s.split(","):
            item = item.strip()
            if item.startswith(f"{attr}="):
                # split apenas na primeira '='
                val = item.split("=", 1)[1]
                vals.add(val)
    return sorted(vals)

def cell_to_set(x):
    """Converte célula como 'a=1,b=2' em set({'a=1','b=2'}). Trata NaN e strings vazias."""
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return {it.strip() for it in str(x).split(",") if it.strip()}


# Função para transformar string "a=x,b=y" em set {"a=x","b=y"}
def str_para_set(s):
    if pd.isna(s) or s == "":
        return set()
    return {x.strip() for x in str(s).split(",") if x.strip()}


def parse_val(s):
    return s.split("=", 1)[1] if "=" in s else s


def parse_um_item(side_str):
    """'attr=val' -> (attr, val). Se tiver vírgula (mais de 1 item), retorna (None, None)."""
    itens = [s.strip() for s in str(side_str).split(',') if s.strip()]
    if len(itens) != 1:
        return (None, None)
    item = itens[0]
    if '=' in item:
        a, v = item.split('=', 1)
        return a.strip(), v.strip()
    return item.strip(), None


def _split_items_cell(cell):
    """
    Converte a célula (vários formatos) numa lista de itens do tipo 'attr=val' ou 'attr_val'.
    Tratamentos:
      - set / frozenset / list / tuple
      - strings com "frozenset{...}" ou "set{...}"
      - strings com parênteses "('a_b','c_d')"
      - strings "a=1,b=2" -> ['a=1','b=2']
      - strings "a_b" -> ['a_b']
    Retorna lista de strings (já stripadas).
    """
    if cell is None:
        return []
    # se já for coleção (set, list, tuple)
    if isinstance(cell, (set, frozenset, list, tuple)):
        return [str(x).strip().strip("'\"") for x in cell]

    s = str(cell).strip()

    # frozenset({'a_b','c_d'}) ou set({...})
    if s.startswith("frozenset") or s.startswith("set"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = s[start+1:end]
            return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]

    # parênteses: ('a_b', 'c_d')
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]

    # caso normal: "a=1,b=2" ou "a_b"
    return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]


def _normalize_item_to_attr_eq_val(item):
    """
    Tenta transformar 'attr_val' ou 'attr=val' em 'attr=val'.
    Se não for possível, devolve a string original (trimmed).
    """
    it = str(item).strip()
    # já no formato attr=val
    if "=" in it:
        return it.replace(" ", "")   # tira espaços que atrapalham
    # se for attr_val -> transformar primeiro '_' encontrado em '='
    if "_" in it:
        return it.replace("_", "=", 1)
    # fallback
    return it


import pandas as pd

import pandas as pd

def filtrar_regras_por_atributo_exato(df_regras, regras_usuario):
    import pandas as pd
    import itertools

    linhas_filtradas = []

    for regra in regras_usuario:
        ant_attrs = regra["antecedente"]
        cons_attrs = regra["consequente"]
        if isinstance(ant_attrs, str):
            ant_attrs = [ant_attrs]
        if isinstance(cons_attrs, str):
            cons_attrs = [cons_attrs]

        # Pega todos os valores possíveis na base para cada atributo
        def valores_possiveis(col, attrs):
            res = []
            for attr in attrs:
                vals = sorted({
                    item.split("=", 1)[1]
                    for cel in df_regras[col].dropna()
                    for item in str(cel).split(",")
                    if item.startswith(f"{attr}=")
                })
                res.append(vals)
            return res

        ant_vals_list = valores_possiveis("antecedente", ant_attrs)
        cons_vals_list = valores_possiveis("consequente", cons_attrs)

        ant_combs = list(itertools.product(*ant_vals_list))
        cons_combs = list(itertools.product(*cons_vals_list))

        def str_to_set(s):
            if pd.isna(s) or s.strip() == "":
                return set()
            return {x.strip() for x in s.split(",")}

        for a_comb in ant_combs:
            for c_comb in cons_combs:
                regra_ant = ",".join(f"{attr}={val}" for attr, val in zip(ant_attrs, a_comb))
                regra_cons = ",".join(f"{attr}={val}" for attr, val in zip(cons_attrs, c_comb))

                set_ant = str_to_set(regra_ant)
                set_cons = str_to_set(regra_cons)

                # Filtra regras do mesmo tamanho (simples ou composta)
                df_temp = df_regras[
                    df_regras["antecedente"].apply(lambda x: str_to_set(x) == set_ant) &
                    df_regras["consequente"].apply(lambda x: str_to_set(x) == set_cons)
                ]

                if not df_temp.empty:
                    linhas_filtradas.append(df_temp.iloc[0])

    if linhas_filtradas:
        return pd.DataFrame(linhas_filtradas).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df_regras.columns)

def filtrar_regras_por_atributo_valores(df_regras, regras_usuario, valores_possiveis):
    """
    Expande atributos escolhidos pelo usuário em todas as combinações
    de valores possíveis e filtra df_regras para manter apenas essas.
    
    df_regras: DataFrame com colunas ["antecedente","consequente",...]
    regras_usuario: lista de dicts [{"antecedente": "first_pull", "consequente": "merged"}, ...]
    valores_possiveis: dict {"atributo": ["val1","val2",...], ...}
    """
    regras_finais = []
    
    for r in regras_usuario:
        ant_attr = r["antecedente"]
        cons_attr = r["consequente"]
        
        # valores possíveis do antecedente e consequente
        ant_vals = valores_possiveis.get(ant_attr, [])
        cons_vals = valores_possiveis.get(cons_attr, [])
        
        for av in ant_vals:
            for cv in cons_vals:
                regra_ant = f"{ant_attr}={av}"
                regra_cons = f"{cons_attr}={cv}"
                
                # filtrar no df_regras
                match = df_regras[
                    (df_regras["antecedente"] == regra_ant) &
                    (df_regras["consequente"] == regra_cons)
                ]
                if not match.empty:
                    regras_finais.append(match)

    if regras_finais:
        return pd.concat(regras_finais, ignore_index=True)
    else:
        return pd.DataFrame(columns=df_regras.columns)

def listar_valores_por_atributo(df, top=20):
    """
    Percorre o DataFrame sem alterar nada e retorna os valores únicos de cada atributo.
    """
    resultados = {}
    for col in df.columns:
        try:
            valores = df[col].dropna().unique().tolist()
        except Exception:
            valores = []
        if len(valores) > top:
            valores = valores[:top] + ["..."]
        resultados[col] = valores
    return resultados

def calcular_meses(data_inicio, data_fim):
    """Calcula o número de meses entre duas datas, incluindo o mês inicial."""
    return (data_fim.year - data_inicio.year) * 12 + (data_fim.month - data_inicio.month)

def particionar_por_meses(df, col_data, meses_por_particao):
    """
    Divide o DataFrame em partições consecutivas de X meses.
    Retorna uma lista de dicionários com cada partição e suas informações:
    'df', 'data_min', 'data_max', 'meses'.
    """
    particoes = []
    #df = df.sort_values(col_data)
    data_inicio = df[col_data].min()
    data_fim = df[col_data].max()

    while data_inicio <= data_fim:
        # Não subtraímos 1 dia aqui para evitar erro de arredondamento
        data_final_part = data_inicio + pd.DateOffset(months=meses_por_particao)
        df_part = df[(df[col_data] >= data_inicio) & (df[col_data] < data_final_part)]
        
        if not df_part.empty:
            particoes.append({
                "data_min": df_part[col_data].min(),
                "data_max": df_part[col_data].max(),
                "dados": df_part,
                #"meses": calcular_meses(df_part[col_data].min(), df_part[col_data].max())
            })
        
        data_inicio = data_final_part

    return particoes

def particionar_por_marcos(df, col_data, marcos):
    """
    Divide o DataFrame com base em marcos temporais definidos pelo usuário.
    'marcos' deve ser uma lista de datas (datetime).
    Retorna uma lista de dicionários com cada partição e suas informações.
    """
    df = df.sort_values(col_data)
    particoes = []
    data_inicio = df[col_data].min()

    # Garantir que os marcos estejam ordenados
    marcos = sorted(marcos)

    for marco in marcos:
        df_part = df[(df[col_data] >= data_inicio) & (df[col_data] < marco)]
        if not df_part.empty:
            particoes.append({
                "df": df_part,
                "data_min": df_part[col_data].min(),
                "data_max": df_part[col_data].max()
            })
        data_inicio = marco

    # Última partição (após o último marco)
    df_part_final = df[(df[col_data] >= data_inicio)]
    if not df_part_final.empty:
        particoes.append({
            "df": df_part_final,
            "data_min": df_part_final[col_data].min(),
            "data_max": df_part_final[col_data].max()
        })

    return particoes

def particionar_por_registros(df, tamanho_particao, col_data):
    """
    Divide o DataFrame em partições de tamanho fixo (número de registros).
    Retorna uma lista de dicionários com cada partição e suas informações.
    
    Parâmetros:
    - df: DataFrame original
    - tamanho_particao: número de registros por partição
    - col_data: nome da coluna de data (usada para calcular data_min e data_max)
    """
    df = df.sort_values(col_data).reset_index(drop=True)
    particoes = []

    total_registros = len(df)
    num_particoes = (total_registros // tamanho_particao) + (1 if total_registros % tamanho_particao else 0)

    for i in range(0, total_registros, tamanho_particao):
        df_part = df.iloc[i:i + tamanho_particao]
        if not df_part.empty:
            particoes.append({
                "data_min": df_part[col_data].min(),
                "data_max": df_part[col_data].max(),
                "dados": df_part,
            })

    return particoes

def particionar_por_quantidade_igual(df, n_particoes, col_data):
    """
    Divide o DataFrame em n_particoes aproximadamente iguais em número de registros,
    distribuindo registros excedentes nas últimas partições.

    Exemplo: se 10 registros para 3 partições => [3, 3, 4]
    """

    n_total = len(df)
    if n_total == 0:
        return []

    n_particoes = min(n_particoes, n_total)
    base_size = n_total // n_particoes
    resto = n_total % n_particoes

    particoes = []
    inicio = 0

    for i in range(n_particoes):
        fim = inicio + base_size + (1 if i >= n_particoes - resto else 0)
        df_part = df.iloc[inicio:fim].copy()
        if not df_part.empty:
            particoes.append({
                "data_min": df_part[col_data].min(),
                "data_max": df_part[col_data].max(),
                "dados": df_part
            })
        inicio = fim

    return particoes


def particionar_por_tempo_equal_length(df, col_data, n_particoes):
    """
    Particiona o DataFrame em n_particoes com intervalos de tempo iguais.
    Retorna uma lista de dicionários com 'data_min', 'data_max' e 'dados'.
    
    Parâmetros:
        df (pd.DataFrame): base de dados completa
        col_data (str): nome da coluna de data
        n_particoes (int): número de partições desejadas
        
    Retorna:
        list[dict]: lista com as partições e seus intervalos
    """
    # Garante que a coluna está em formato datetime
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    df = df.dropna(subset=[col_data])

    # Ordena os dados pela data
    df = df.sort_values(by=col_data).reset_index(drop=True)

    # Define intervalo total de tempo
    data_min_global = df[col_data].min()
    data_max_global = df[col_data].max()

    # Calcula o intervalo de tempo para cada partição
    duracao_total = data_max_global - data_min_global
    duracao_particao = duracao_total / n_particoes

    particoes = []
    inicio = data_min_global

    # Cria partições
    for i in range(n_particoes):
        fim = inicio + duracao_particao

        # Última partição inclui o limite máximo
        if i == n_particoes - 1:
            df_part = df[(df[col_data] >= inicio) & (df[col_data] <= data_max_global)]
        else:
            df_part = df[(df[col_data] >= inicio) & (df[col_data] < fim)]

        if not df_part.empty:
            particoes.append({
                "data_min": df_part[col_data].min(),
                "data_max": df_part[col_data].max(),
                "dados": df_part
            })

        inicio = fim

    return particoes