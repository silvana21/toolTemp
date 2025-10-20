import pandas as pd
from test_r import gerar_regras_com_r_via_subprocess

# Criar DataFrame de exemplo
data = {
    "A": ["sim", "não", "sim", "sim", "não", "sim", "não", "sim", "não", "sim"],
    "B": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
    "C": ["verde", "verde", "azul", "azul", "verde", "azul", "verde", "azul", "verde", "azul"]
}
df = pd.DataFrame(data)

# Chamar função que gera regras
df_regras = gerar_regras_com_r_via_subprocess(df, script_path="gerar_regras.R")

# Mostrar resultados
print(df_regras)