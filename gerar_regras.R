# gerar_regras.R
args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]
sup <- as.numeric(args[3])
conf <- as.numeric(args[4])
lhs_attr <- args[5]
rhs_attr <- args[6]

suppressMessages({
  library(arules)
  library(dplyr)
})

#Ler e preparar dados
df <- read.csv(input_csv, stringsAsFactors = FALSE)
#Limpeza...
df[] <- lapply(df, function(x) factor(as.character(trimws(gsub('"', '', x)))))
transacoes <- as(df, "transactions")

# Selecionar todos os itens do atributo informado
lhs_items <- if (lhs_attr != "") grep(paste0("^", lhs_attr, "="), itemLabels(transacoes), value = TRUE) else NULL
rhs_items <- if (rhs_attr != "") grep(paste0("^", rhs_attr, "="), itemLabels(transacoes), value = TRUE) else NULL

regras <- apriori(
  transacoes,
  parameter = list(supp = sup, conf = conf, minlen = 2),
  appearance = list(lhs = lhs_items, rhs = rhs_items)
)

# Converter em data.frame
if (length(regras) == 0) {
  df_regras <- data.frame()
} else {
  df_regras <- as(regras, "data.frame")
  df_regras$antecedente <- sapply(strsplit(as.character(df_regras$rules), " => "), `[`, 1)
  df_regras$consequente <- sapply(strsplit(as.character(df_regras$rules), " => "), `[`, 2)
  df_regras <- df_regras[, c("antecedente", "consequente", "support", "confidence", "lift")]
}

write.csv(df_regras, output_csv, row.names = FALSE)