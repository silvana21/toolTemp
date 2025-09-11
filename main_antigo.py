from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import pandas as pd

app = FastAPI()

global_df = None  # Guarda o DataFrame carregado para usar depois

def render_page(content_html: str) -> str:
    return f"""
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            margin: 40px;
            color: #333;
          }}
          h1 {{
            color: #2c3e50;
          }}
          form {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            max-width: 600px;
            margin-bottom: 30px;
          }}
          input[type="submit"] {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
          }}
          input[type="submit"]:hover {{
            background-color: #2980b9;
          }}
          table {{
            border-collapse: collapse;
            width: 100%;
            max-width: 600px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }}
          th, td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
          }}
          th {{
            background-color: #3498db;
            color: white;
          }}
          tr:hover {{
            background-color: #f1f1f1;
          }}
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
    df = pd.read_csv(file.file)
    global_df = df
    
    datetime_cols = []
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            ratio = converted.notna().mean()
            if ratio > 0.7:
                datetime_cols.append(col)
        except Exception:
            continue

    columns_html = "<br>".join(df.columns)
    head_html = df.head().to_html(index=False)

    if not datetime_cols:
        content = "<h2>Erro: Nenhuma coluna de data detectada no arquivo.</h2>"
        return render_page(content)
    
    if len(datetime_cols) == 1:
        col_data = datetime_cols[0]
        content = f"""
        <h1>Coluna Data Detectada</h1>
        <p>Foi identificada a coluna <b>{col_data}</b> como coluna de data.</p>
        <p>Esta coluna será momentaneamente descartada para gerar as regras de associação.</p>
        <h2>Colunas do arquivo original:</h2>
        <p>{columns_html}</p>
        <h2>Primeiras 5 linhas:</h2>
        {head_html}
        <form action="/confirm_data_columns" method="post">
            <input type="hidden" name="selected_cols" value="{col_data}">
            <input type="submit" value="Confirmar e continuar">
        </form>
        """
        return render_page(content)
    else:
        options_html = ""
        for c in datetime_cols:
            options_html += f'<input type="checkbox" name="selected_cols" value="{c}" checked> {c}<br>'
        
        content = f"""
        <h1>Confirmação das Colunas Data</h1>
        <p>Foram identificadas as seguintes colunas como datas. Selecione as que deseja considerar para particionamento:</p>
        <form action="/confirm_data_columns" method="post">
            {options_html}
            <input type="submit" value="Confirmar seleção">
        </form>
        <h2>Colunas do arquivo original:</h2>
        <p>{columns_html}</p>
        <h2>Primeiras 5 linhas:</h2>
        {head_html}
        """
        return render_page(content)

@app.post("/confirm_data_columns", response_class=HTMLResponse)
async def confirm_data_columns(selected_cols: list = Form(...)):
    global global_df
    if isinstance(selected_cols, str):
        selected_cols = [selected_cols]
    
    df_sem_data = global_df.drop(columns=selected_cols)

    columns_html = "<br>".join(global_df.columns)
    head_html = global_df.head().to_html(index=False)

    content = f"""
    <h2>Colunas removidas para mineração:</h2>
    <p>{', '.join(selected_cols)}</p>
    <h2>Colunas do arquivo original:</h2>
    <p>{columns_html}</p>
    <h2>Primeiras 5 linhas do arquivo original:</h2>
    {head_html}
    <p>Aqui você poderá seguir para a próxima etapa, como mineração e análise temporal.</p>
    """
    return render_page(content)
