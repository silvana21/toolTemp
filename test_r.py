import subprocess
import pandas as pd
import tempfile
import os

def gerar_regras_com_r_via_subprocess(df, script_path="gerar_regras.R"):
    # criar arquivos temporários
    input_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    # fechar os arquivos para que o R possa ler/escrever
    input_csv.close()
    output_csv.close()

    # Convertendo todas as colunas para string, removendo espaços e aspas
    df = df.astype(str).apply(lambda x: x.str.strip().replace('"', ''))

    # salvar CSV temporário
    df.to_csv(input_csv.name, index=False)

    result = subprocess.run(
        ["Rscript", script_path, input_csv.name, output_csv.name]
        #,
        #text=True
    )

    if result.returncode != 0:
        print("=== ERRO DO R ===")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Erro ao executar Rscript")

    # ler CSV gerado pelo R
    df_regras = pd.read_csv(output_csv.name)

    # remover arquivos temporários
    os.unlink(input_csv.name)
    os.unlink(output_csv.name)

    return df_regras