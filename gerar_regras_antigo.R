# gerar_regras.R
# Recebe CSV de entrada e gera CSV de regras

args <- commandArgs(trailingOnly=TRUE)
input_csv <- args[1]   # caminho do CSV de entrada
output_csv <- args[2]  # caminho do CSV de saída

library(arules)

# Ler dados
df <- read.csv(input_csv, stringsAsFactors = FALSE)
# Remover espaços e aspas de valores
df[] <- lapply(df, function(x) {
  x <- trimws(gsub('"', '', x))
  x[x == "NA" | x == ""] <- NA
  return(x)
})

df[] <- lapply(df, function(x) factor(as.character(x)))

# converte direto para transactions (cada linha -> transação com N itens)
transacoes <- as(df, "transactions")

# Gerar regras
regras <- apriori(transacoes, parameter = list(supp=0.01, conf=0.01, minlen=2))

# Converter para data.frame
df_regras <- as(regras, "data.frame")

# Verificar se existem regras
if (nrow(df_regras) == 0) {
  df_regras_final <- data.frame(
    antecedente=character(0),
    consequente=character(0),
    support=numeric(0),
    confidence=numeric(0),
    lift=numeric(0)
  )
} else {
  # Separar antecedente e consequente
  df_regras$antecedente <- sapply(strsplit(as.character(df_regras$rules), " => "), `[`, 1)
  df_regras$consequente <- sapply(strsplit(as.character(df_regras$rules), " => "), `[`, 2)

  # Selecionar colunas de interesse (suporte/confiança/lift podem não existir se nenhuma regra)
  cols_existentes <- intersect(
    c("antecedente", "consequente", "support", "confidence", "lift"),
    colnames(df_regras)
  )
  df_regras_final <- df_regras[, cols_existentes, drop=FALSE]
}

# Salvar CSV
write.csv(df_regras_final, output_csv, row.names = FALSE)