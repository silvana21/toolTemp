# Usar Python 3.10 slim
FROM python:3.10-slim

# Criar e usar pasta da aplicação
WORKDIR /app

# Copiar arquivos para dentro do container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para rodar o FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
