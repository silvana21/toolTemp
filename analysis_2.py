# analysis.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import io
import matplotlib.pyplot as plt

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
    df_regras['antecedente'] = df_regras['antecedents'].apply(lambda x: ','.join(list(x)))
    df_regras['consequente'] = df_regras['consequents'].apply(lambda x: ','.join(list(x)))
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
    import pandas as pd

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

def filtrar_regras_por_atributo(df_regras, regras_usuario):
    """
    Filtra regras do DataFrame de acordo com as regras completas selecionadas pelo usuário.
    Cada regra do usuário é do tipo: {'antecedente': 'atributo', 'consequente': 'atributo'}.

    - df_regras: DataFrame com colunas ['antecedente', 'consequente', 'suporte', 'confianca', 'lift']
    - regras_usuario: lista de dicts, cada dict com 'antecedente' e 'consequente' (somente atributos)

    Retorna: DataFrame filtrado contendo todas as regras correspondentes.
    """
    import pandas as pd

    if not regras_usuario or df_regras.empty:
        return pd.DataFrame(columns=df_regras.columns)

    filtros = []
    for regra in regras_usuario:
        atr_ant = regra["antecedente"]
        atr_cons = regra["consequente"]

        # Seleciona todas as regras cujo antecedente contenha qualquer rótulo do atributo escolhido
        # e consequente contenha qualquer rótulo do atributo escolhido
        filtro = df_regras[
            df_regras['antecedente'].str.startswith(f"{atr_ant}=") &
            df_regras['consequente'].str.startswith(f"{atr_cons}=")
        ]
        filtros.append(filtro)

    if filtros:
        df_filtrado = pd.concat(filtros, ignore_index=True).drop_duplicates()
    else:
        df_filtrado = pd.DataFrame(columns=df_regras.columns)

    return df_filtrado



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
