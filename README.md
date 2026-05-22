# Temporal Association Rule Analyzer

Ferramenta web para análise temporal de regras de associação em dados de Engenharia de Software.

## Sobre a Ferramenta

A ferramenta foi desenvolvida para apoiar análises temporais de regras de associação extraídas de dados provenientes de repositórios de software, especialmente dados relacionados a *pull requests* de projetos de código aberto.

A aplicação permite extrair regras de associação utilizando o algoritmo Apriori e acompanhar a evolução temporal de medidas de interesse como Suporte, Confiança e Lift em diferentes partições temporais da base de dados.

O objetivo é auxiliar pesquisadores e profissionais na identificação de mudanças temporais em padrões observados em ambientes colaborativos de desenvolvimento de software.

---

## Principais Funcionalidades

- Importação de bases de dados em formato CSV;
- Extração de regras de associação utilizando Apriori;
- Configuração de suporte e confiança mínimos;
- Visualização das medidas:
  - Suporte;
  - Confiança;
  - Lift;
- Particionamento temporal da base de dados;
- Definição manual de marcos temporais;
- Comparação temporal das regras de associação;
- Visualização gráfica da evolução temporal das medidas;
- Análise temporal de antecedentes e consequentes das regras.

---

## Estratégias de Particionamento Temporal

Atualmente, a ferramenta suporta três estratégias de particionamento:

1. Definição manual de marcos temporais;
2. Intervalos temporais de mesmo tamanho;
3. Particionamento com mesma quantidade de registros.

---

## Tecnologias Utilizadas

- Python
- Streamlit
- MLxtend
- Pandas
- Plotly
- Matplotlib

Bibliotecas adicionais:

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from dateutil.relativedelta import relativedelta
```
Instalação

Clone o repositório:

git clone https://github.com/silvana21/toolTemp.git

Acesse a pasta do projeto:

cd SEU-REPOSITORIO

Instale as dependências:

pip install -r requirements.txt

Execução

Para iniciar a aplicação:
```
streamlit run app.py
```
Formato dos Dados

A ferramenta espera arquivos CSV contendo atributos categóricos.

Atributos contínuos e temporais devem ser previamente discretizados quando necessário para participação na mineração das regras de associação.

Exemplos de Uso

A ferramenta pode ser utilizada em cenários como:

análise temporal de aceitação de pull requests;
investigação de padrões de contribuição;
estudos sobre evolução de práticas colaborativas;
mineração de repositórios de software;
estudos empíricos em Engenharia de Software.
Licença

Este projeto está licenciado sob a licença MIT.

Contato

Silvana de Andrade Gonçalves
Universidade Federal Fluminense (UFF)
