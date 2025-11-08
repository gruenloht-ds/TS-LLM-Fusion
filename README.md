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
├── data/                # Store datasets
├── models/              # Model architectures or wrappers
├── experiments/         # Training scripts, evaluation
└── src/                 # Helper functions, preprocessing, custom modules, etc.
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

```bash
wget ...
```
