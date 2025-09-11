# Nova lógica incorporada ao processo de análise geral usando mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from openpyxl import Workbook
from collections import defaultdict

# Nova lógica incorporada ao processo de análise geral usando mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from openpyxl import Workbook

# Função para preparar os dados, removendo atributos numéricos e de data
def preparar_dados_para_mineracao(csv_path):
    df_original = pd.read_csv(csv_path)

    atributos_numericos = []
    atributos_de_data = []

    for coluna in df_original.columns:
        if pd.api.types.is_numeric_dtype(df_original[coluna]) and df_original[coluna].nunique() > 10:
            atributos_numericos.append(coluna)
        elif pd.to_datetime(df_original[coluna], errors='coerce').notna().sum() > len(df_original) * 0.9:
            atributos_de_data.append(coluna)

    atributos_remover = atributos_numericos + atributos_de_data
    dados_removidos = df_original[atributos_remover].copy()
    df_discretizado = df_original.drop(columns=atributos_remover)

    # Converte valores booleanos para string "True"/"False"
    for col in df_discretizado.select_dtypes(include=['bool']).columns:
        df_discretizado[col] = df_discretizado[col].astype(str)

    return df_discretizado, dados_removidos, atributos_remover

# Função para reintegrar atributos ao final
def reincorporar_atributos(df, dados_removidos):
    return pd.concat([df.reset_index(drop=True), dados_removidos.reset_index(drop=True)], axis=1)

# Função para transformar dados discretizados em formato one-hot para mlxtend
def preparar_para_apriori(df):
    df_oht = pd.get_dummies(df, prefix_sep='=')
    return df_oht

# Função para gerar regras com mlxtend, adicionando métricas de validação manual
def gerar_regras_com_mlxtend(df, sup, conf):
    df_oht = preparar_para_apriori(df)
    frequent_itemsets = apriori(df_oht, min_support=float(sup), use_colnames=True)
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=float(conf))

    df_regras = regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    df_regras['antecedente'] = df_regras['antecedents'].apply(lambda x: ','.join(list(x)))
    df_regras['consequente'] = df_regras['consequents'].apply(lambda x: ','.join(list(x)))
    df_regras.rename(columns={'support': 'suporte', 'confidence': 'confianca'}, inplace=True)
    df_regras['frequencia_absoluta'] = df_regras['suporte'] * len(df_oht)
    df_regras['frequencia_condicional'] = df_regras['confianca'] * df_regras['frequencia_absoluta'] / df_regras['suporte']

    df_regras = df_regras[['antecedente', 'consequente', 'suporte', 'confianca', 'lift', 'frequencia_absoluta', 'frequencia_condicional']]
    return df_regras

# Função atualizada para exportar as regras separando corretamente os atributos booleanos e categóricos
def exportar_regras_para_excel(df_regras, nome_arquivo):
    wb = Workbook()
    aba_padrao = wb.active
    criou_alguma_aba = False

    def extrair_atributo_valor(item):
        if '=' in item:
            partes = item.rsplit('=', 1)
            return partes[0], partes[1]
        return None, None

    # Filtra apenas regras com antecedente e consequente únicos
    df_regras = df_regras[
        (df_regras['antecedente'].apply(lambda x: len(x.split(',')) == 1)) &
        (df_regras['consequente'].apply(lambda x: len(x.split(',')) == 1))
    ]

    # Agrupa regras válidas
    agrupado = defaultdict(list)
    for _, row in df_regras.iterrows():
        ant_attr, ant_val = extrair_atributo_valor(row['antecedente'])
        cons_attr, cons_val = extrair_atributo_valor(row['consequente'])

        if all([ant_attr, ant_val, cons_attr, cons_val]):
            chave = (ant_attr.strip(), ant_val.strip(), cons_attr.strip())
            agrupado[chave].append((cons_val.strip(), row['suporte'], row['confianca'], row['lift']))

    for (ant_attr, ant_val, cons_attr), regras in agrupado.items():
        aba_nome = f"{ant_attr}={ant_val}->{cons_attr}"[:31]  # Excel limita a 31 caracteres
        ws = wb.create_sheet(title=aba_nome)
        ws.append(['Valor do Consequente', 'Suporte', 'Confiança', 'Lift'])

        vistos = set()
        for cons_val, suporte, confianca, lift in regras:
            if cons_val in vistos:
                continue  # evita duplicatas
            ws.append([cons_val, round(suporte, 5), round(confianca, 5), round(lift, 5)])
            vistos.add(cons_val)

        criou_alguma_aba = True

    if criou_alguma_aba:
        wb.remove(aba_padrao)
    else:
        aba_padrao.title = "Aviso"
        aba_padrao.append(["Nenhuma regra foi gerada com os parâmetros fornecidos."])

    wb.save(nome_arquivo)
    print(f"\n✅ Arquivo Excel salvo como: {nome_arquivo}")


#-------------------------------------------------------------------------------------------------
# Função principal para análise geral
def analiseGeral(arq, sup, conf):
    csv_path = 'C:\\Users\\darka\\Codigos\\'+arq+'\\'+arq+'_comData.csv'
    df_discretizado, dados_removidos, _ = preparar_dados_para_mineracao(csv_path)
    df_regras = gerar_regras_com_mlxtend(df_discretizado, sup, conf)

    # Exportar resultado para Excel com múltiplas abas
    exportar_regras_para_excel(df_regras, "PadroesGerais.xlsx")
    print(df_regras.head())
    print("Regras geradas e salvas com sucesso!")
    
#--------------------------------------------------------------------------------------------------
# Nova função de análise particionada unificada
# Nova função de análise particionada unificada
def analiseParticionadaUnificada(arq, sup, conf, modo="automatico", janela=6, intervalo=3, marcos=None):
    csv_path_com_data = 'C:\\Users\\darka\\Codigos\\'+arq+'\\'+arq+'_comData.csv'
    df = pd.read_csv(csv_path_com_data)
    df['created_at'] = pd.to_datetime(df['created_at'])

    nome_base = os.path.splitext(os.path.basename(csv_path_com_data))[0].replace('_comData','')
    pasta_base = os.path.join(os.getcwd(), nome_base)
    os.makedirs(pasta_base, exist_ok=True)

    if modo == "manual":
        if not marcos or len(marcos) < 2:
            raise ValueError("Modo manual exige ao menos dois marcos temporais")
        limites = marcos + [df['created_at'].max().strftime('%Y-%m-%d %H:%M:%S')]
        particoes = [(marcos[i], limites[i+1]) for i in range(len(marcos))]
    else:
        comeca = df['created_at'].min()
        termina = df['created_at'].max()
        particoes = []
        while comeca < termina:
            fim = comeca + pd.DateOffset(months=janela)
            particoes.append((comeca.strftime('%Y-%m-%d %H:%M:%S'), min(fim, termina).strftime('%Y-%m-%d %H:%M:%S')))
            comeca = comeca + pd.DateOffset(months=intervalo)

    for i, (inicio, fim) in enumerate(particoes, start=1):
        df_part = df.query(f"created_at >= '{inicio}' and created_at <= '{fim}'").drop(columns=['created_at'])

        pasta_part = os.path.join(pasta_base, f"p{i}")
        os.makedirs(pasta_part, exist_ok=True)

        part_path = os.path.join(pasta_part, f"particao_{i}.csv")
        df_part.to_csv(part_path, index=False)

        df_discretizado, _, _ = preparar_dados_para_mineracao(part_path)
        df_regras = gerar_regras_com_mlxtend(df_discretizado, sup, conf)

        nome_excel = os.path.join(pasta_part, f"Padroes_Particao_{i}.xlsx")
        exportar_regras_para_excel(df_regras, nome_excel)
        print(f"🔍 Partição {i} ({inicio} a {fim}) processada em {pasta_part}.")

def gerar_regras_em_df(df, sup, conf):
    df_discretizado = preparar_para_apriori(df)
    regras = gerar_regras_com_mlxtend(df_discretizado, sup, conf)
    # Aqui pode vir seu código para filtrar/agrupar se quiser (ou já fazer isso em gerar_regras_com_mlxtend)
    return regras
