# Usar Python 3.12 slim
FROM python:3.12-slim

# Criar e usar pasta da aplicação
WORKDIR /app

# Copiar arquivos para dentro do container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "main_com_mlxtend.py", "--server.address=0.0.0.0", "--server.port=8501"]
