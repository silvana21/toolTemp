# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import pandas as pd
import io

import analysis  # seu arquivo analysis.py ao lado

app = FastAPI()

# armazenamento em memória (protótipo)
global_df = None
global_df_sem_data = None
global_selected_data_cols = None
global_last_rules_df = None

def render_page(content_html: str) -> str:
    # (copie aqui sua função render_page com CSS)
    return f"""
    <html>
      <head>
        <style>
          /* seu CSS aqui (mantive curto para exemplo) */
          body {{ font-family: Arial, sans-serif; background:#f9f9f9; margin:40px; color:#333; }}
          h1 {{ color:#2c3e50; }}
          form {{ background:white; padding:20px; border-radius:8px; max-width:800px; }}
          input[type=submit] {{ background:#3498db; color:white; padding:8px 16px; border:none; border-radius:4px; cursor:pointer; }}
          table {{ border-collapse:collapse; width:100%; }}
          th, td {{ padding:8px; border:1px solid #ddd; }}
          th {{ background:#3498db; color:white; }}
        </style>
      </head>
      <body>
        <h1>Minha Ferramenta de Análise Temporal</h1>
        {content_html}
      </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def read_root():
    content = """
    <h2>Upload do arquivo CSV</h2>
    <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept=".csv" required>
        <input type="submit" value="Enviar">
    </form>
    """
    return render_page(content)

@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(file: UploadFile = File(...)):
    global global_df
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    global_df = df

    # Detecta colunas data automaticamente
    datetime_cols = []
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            ratio = converted.notna().mean()
            if ratio > 0.7:
                datetime_cols.append(col)
        except Exception:
            pass

    columns_html = "<br>".join(df.columns)
    head_html = df.head().to_html(index=False)

    if not datetime_cols:
        content = "<h2>Erro: Nenhuma coluna de data detectada no arquivo.</h2>"
        return render_page(content)

    if len(datetime_cols) == 1:
        # se só 1, informamos e criamos df_sem_data automaticamente (sem confirmação)
        col_data = datetime_cols[0]
        # remove colunas de data e guarda
        df_sem_data, dados_removidos, removed_cols = analysis.preparar_dados_para_mineracao_from_df(df)
        # Guarda em memória
        global global_df_sem_data, global_selected_data_cols
        global_df_sem_data = df_sem_data
        global_selected_data_cols = removed_cols  # lista
        content = f"""
        <h2>Coluna data detectada: <b>{col_data}</b></h2>
        <p>Essa(s) coluna(s) será(ão) removida(s) temporariamente para mineração.</p>
        <h3>Primeiras 5 linhas do arquivo:</h3>
        {head_html}
        <form action="/run_mining" method="post">
            <label>Min Support (ex: 0.01): <input name="min_support" value="0.01" /></label><br>
            <label>Min Confidence (ex: 0.5): <input name="min_confidence" value="0.5" /></label><br><br>
            <input type="submit" value="Gerar Regras">
        </form>
        """
        return render_page(content)

    else:
        # mais de uma data: pedir ao usuário para selecionar
        options_html = ""
        for c in datetime_cols:
            options_html += f'<input type="checkbox" name="selected_cols" value="{c}" checked> {c}<br>'
        content = f"""
        <h2>Colunas detectadas como data — selecione:</h2>
        <form action="/confirm_data_columns" method="post">
            {options_html}
            <input type="submit" value="Confirmar seleção">
        </form>
        <h3>Primeiras 5 linhas do arquivo:</h3>
        {head_html}
        """
        return render_page(content)

@app.post("/confirm_data_columns", response_class=HTMLResponse)
async def confirm_data_columns(selected_cols: list = Form(...)):
    """
    Recebe lista de colunas de data selecionadas (checkboxes).
    Prepara df_sem_data e apresenta formulário para parâmetros de mineração.
    """
    global global_df, global_df_sem_data, global_selected_data_cols
    if isinstance(selected_cols, str):
        selected_cols = [selected_cols]

    # removemos as colunas selecionadas (pode remover também numéricos no preparar)
    df_sem_data = global_df.drop(columns=selected_cols)
    # opcional: aplicar a função de preparar_dados para remover numericos também
    df_sem_data, dados_removidos, removed_cols_extra = analysis.preparar_dados_para_mineracao_from_df(df_sem_data)
    # guardamos
    global_df_sem_data = df_sem_data
    global_selected_data_cols = selected_cols + removed_cols_extra

    head_html = global_df.head().to_html(index=False)
    content = f"""
    <h2>Colunas selecionadas para remoção:</h2>
    <p>{', '.join(selected_cols)}</p>
    <h3>Primeiras 5 linhas do arquivo original:</h3>
    {head_html}
    <form action="/run_mining" method="post">
        <label>Min Support (ex: 0.01): <input name="min_support" value="0.01" /></label><br>
        <label>Min Confidence (ex: 0.5): <input name="min_confidence" value="0.5" /></label><br><br>
        <input type="submit" value="Gerar Regras">
    </form>
    """
    return render_page(content)


@app.post("/run_mining", response_class=HTMLResponse)
async def run_mining(min_support: float = Form(...), min_confidence: float = Form(...)):
    """
    Executa a mineração usando os parâmetros e o dataframe em memória.
    Mostra as primeiras regras e oferece download do Excel.
    """
    global global_df_sem_data, global_last_rules_df
    if global_df_sem_data is None:
        return render_page("<h2>Erro: Nenhum dado disponível para mineração. Faça upload primeiro.</h2>")

    # chama a função do analysis.py
    df_regras = analysis.gerar_regras_com_mlxtend(global_df_sem_data, sup=min_support, conf=min_confidence)
    global_last_rules_df = df_regras

    if df_regras.empty:
        content = "<h2>Nenhuma regra encontrada com os parâmetros fornecidos.</h2><a href='/'>Voltar</a>"
        return render_page(content)

    regras_html = df_regras.head(50).to_html(index=False)
    content = f"""
    <h2>Regras geradas (exibindo até 50 linhas)</h2>
    {regras_html}
    <form action="/download_rules" method="get">
        <input type="submit" value="Download Excel das Regras">
    </form>
    <br><a href='/'>Novo upload</a>
    """
    return render_page(content)


@app.get("/download_rules")
async def download_rules():
    """
    Gera o Excel em memória e retorna como StreamingResponse para download.
    """
    global global_last_rules_df
    if global_last_rules_df is None:
        return HTMLResponse("<h2>Nenhuma regra disponível para download.</h2>")

    excel_bytes = analysis.exportar_regras_para_excel_bytes(global_last_rules_df)
    return StreamingResponse(io.BytesIO(excel_bytes), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=regras.xlsx"})

