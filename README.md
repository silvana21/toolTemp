## TEMAR - Temporal Association Rule Analyzer

TEMAR is a web-based tool developed to support the temporal analysis of association rules extracted from Software Engineering datasets, especially data obtained from pull requests of open-source software repositories.

The tool applies the Apriori algorithm to discover association rules and allows users to analyze how rule quality measures evolve over time, including:

Support
Confidence
Lift

The system supports multiple temporal partitioning strategies, enabling researchers to investigate how association patterns change during software evolution.


## Features

Features
Import CSV datasets
Mine association rules using Apriori
Configure minimum support and confidence
Temporal partitioning
 Manual temporal milestones
 Equal-size intervals
 Equal-number-of-records partitions
Compare rule evolution over time
Interactive visualizations
Plot Support, Confidence and Lift


## Repository Organization

├── main_com_mlxtend.py            # Streamlit application
├── requirements.txt               # Python dependencies
├── datasets/                      # Example datasets
├── paper/                         # Paper Accepted at SBES 2026
├── analysis.py                    # Streamlit auxiliar functions
├── LICENSE
└── README.md


## Associated Paper

This artifact accompanies the paper:

TEMAR: Uma Ferramenta para Análise Temporal de Regras de Associação em Dados de Repositórios de Software

Accepted at SBES 2026.

[paper](./paper/paper_TEMAR_SBES2026.pdf) 

## Requirements

Python 3.12

Tested on:

- Windows 11

Required libraries:

- numpy==2.3.2
- pandas
- matplotlib
- mlxtend
- plotly
- streamlit
- streamlit-option-men
- streamlit-sortables==0.3.1
- python-dateutil

Hardware requirements

- 4 GB RAM minimum
- 500 MB free disk space

No GPU is required.


## Installation

Clone the repository

```git clone --branch v1.0.2 https://github.com/silvana21/toolTemp.git```

Enter the project

```cd toolTemp```

Create a virtual environment (optional)

```python -m venv .venv```

Activate the environment

Windows

```.venv\Scripts\activate```

Linux/macOS

```source .venv/bin/activate```

Install dependencies

```pip install -r requirements.txt```

Running the Tool

```
streamlit run main_com_mlxtend.py
```

The application should start a local Streamlit server and display a message similar to:

Local URL:
http://localhost:8501

Open the URL in a web browser.


## Input Data

The tool expects CSV files containing categorical attributes.
Continuous and temporal attributes must be discretized beforehand, when necessary, for inclusion in association rule mining.


## Example Workflow

- Run the application.
- Load one of the sample CSV files.
- Configure minimum support and confidence.
- Choose the rules to analyze.
- Run the mining process for the complete dataset.
- Choose the partitioning strategy.
- Configure minimum support and confidence to temporal analysis.
- Run the mining process for the partitioned datasets.
- View the charts.


## Example Datasets

The datasets/ directory contains example datasets extracted from open-source software repositories for demonstrating and testing the TemAR tool.

Each CSV file includes categorical data derived from pull requests, contributors, and code review outcomes. These datasets can be directly loaded through the TemAR interface to explore the tool's temporal association rule mining capabilities.

To use an example dataset:

- Launch the application.
- Select Load Dataset.
- Choose one of the CSV files available in the datasets/ directory.
- Configure the mining parameters and chose the rules to analyze.
- Run the analysis and explore the temporal visualizations presented.


## Technologies

Python
Streamlit
Pandas
MLxtend
Plotly
Matplotlib
Licença


## License

MIT License.


## Citation

This version of the artifact is archived on Zenodo:

DOI: [will be updated after the new Zenodo version is published]


## Author

Silvana de Andrade Gonçalves

Universidade Federal Fluminense (UFF)

Instituto Federal do Acre (IFAC)
