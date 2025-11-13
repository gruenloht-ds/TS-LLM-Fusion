# TS-LLM-Fusion

This project explores how large language-models and transformer architectures can be adapted for time-series reasoning, temporal understanding, and forecasting.

This project is inspired by two recent works:

[OpenTSLM: Time-Series Language Models for Reasoning Over Multivariate Medical Text-and-Time-Series Data](https://arxiv.org/pdf/2510.02410)

[LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics](https://arxiv.org/pdf/2503.09656)

Our goal is to understand, implement, and experiment with ideas from these papers, and to build a small prototype model or pipeline that combines:

- LLM-style semantic understanding
- Time-series pattern extraction
- Forecasting and reasoning over sequential data


## Project Structure
```
TS-LLM-Fusion/
├── configs              # .yaml files specifying architectures and data paths
│   ├── model.yaml
│   └── path.yaml
├── data                 # folder to keep all the real and synthetic data to benchmark
│   ├── real
│   └── synthetic
├── models               # store trained models
├── notebooks            # code for testing
├── results              # results from runs or experiments
├── examples             # examples of how to run submodles built in this project
├── extras               # extra material for this project - path to datasets, dataset citations
├── scripts              # scripts to run - download data, train, evaluate, graph, ...
└── src                  # code to build out this project, dataloaders, architecutres, data generation/pulling
    └── ts_llm_fusion
        ├── core         # loading configs, saving logs
        ├── data         # code for getting the data
        ├── llm_ps       # code for creating modules from the LLM-PS paper
        ├── models       # other model architecutres
        ├── synthetic    # generating synthetic data
        ├── tslm         # code for creating modules from the OpenTSLM paper
        └── utils        # random useful functions
```

## Setup

This section details how to run code in this repository and recreate results.

### Environment

1. First create a python venv, and install the requirements

```bash
python -m venv ~/.venvs/ts_llm
source ~/.venvs/ts_llm
pip install -r requirements.txt
```

2. Install packages in editable model

To make all fucntions and modules inside of `src/` accessible as a python package, run:

```bash
pip install -e .
```

### Download Datasets

... In Progress ...
