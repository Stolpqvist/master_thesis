# Master's Thesis

# Neural network approaches to research grant proposal classification. A comparative study of RNN, CNN, and RoBERTa with environmental impact analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)


> 📄 **Read the full published Master's Thesis:** Read the full published paper on [DiVA Portal / University Archive](https://www.diva-portal.org/smash/get/diva2:2069516/FULLTEXT01.pdf).

This repository contains the official code for our Master's Thesis project conducted in collaboration with **Vetenskapsrådet (The Swedish Research Council - SRC)**. The system automatically classifies research grant proposals using deep learning models and evaluates model performance with rigorous statistical significance testing.



## Authors & Task Ownership

This project was a collaborative research effort. System architecture, data splitting, statistical evaluation, and experiment pipelines were designed jointly, while core model implementations were divided as follows:

* **[Adele](https://github.com/madame-croissant)** ([@madame-croissant](https://github.com/madame-croissant)):
  * **Recurrent Neural Network (RNN)**: Designed, implemented, and tuned the attention-based LSTM model (`model/rnn.py`).
  * **Joint Development**: Built text preprocessing, tokenization, and sequence formatting for neural models. Co-developed the RoBERTa transformer pipeline, data preprocessing workflows, statistical significance testing framework, and experiment orchestration code.

* **[Robin](https://github.com/Stolpqvist)** ([@Stolpqvist](https://github.com/Stolpqvist)):
  * **Convolutional Neural Network (CNN)**: Designed, implemented, and tuned the 4-parallel layer CNN model (`model/cnn.py`).
  * **Joint Development**: Built text preprocessing, tokenization, and sequence formatting for neural models. Co-developed the RoBERTa transformer pipeline, data preprocessing workflows, statistical significance testing framework, and experiment orchestration code..



## Methodology

###  Data
    The data is provided by the SRC and is subject to privacy.
    However, the approach for this data was to split the datasets into k-folds with stratification.

### Models
    - RNN:
        An RNN with an attention-layer

    - CNN:
        A CNN with 4 parallel layers.

    - RoBERTa:
        An xlm-roberta-base model.
### Evaluation
    The code runs a bootstrap against chance and a pairwise comparison. To account for FWER the results
    get corrected with the Bonferroni method.

### Visualisation
    The code allows for the creation of confusion matricies, boxplots and F1-score distribution.
    Visualisation is only performed during bootstrapping.

## Installation

```bash
pip install -r requirements.txt
```

## Usage
To train the model you may:
```bash
python3 main.py --train --model cnn
```
Choose 'cnn', 'rnn', or 'roberta'. Note: training requires access to the SRC dataset, which cannot be shared due to its privacy status.


To test a model:
```bash
python3 main.py --test --model cnn 
```
Note: This also has to be given the -bg flag

The code also incorporates visualisation and emissions tracking:
```bash
python3 main.py --model cnn rnn roberta --boot --vis -em
```

The flag '--boot' enables bootstrapping, '--vis' enables visualisation, and '-em' controls emissions tracking.
### Flags
| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Model(s) to use: `cnn`, `rnn`, `roberta`. Accepts multiple only for `--boot` |
| `--train` | `-tr` | Enable training |
| `--test` | | Enable testing |
| `--boot` | `-b` | Enable bootstrapping |
| `--vis` | `-v` | Enable visualisation |
| `--emissions` | `-em` | Enable emissions tracking |
| `--param_hunt` | `-p` | Enable hyperparameter optimisation |
| `--batch_size` | `-bs` | Batch size (default: 2) |
| `-k` | | Number of k-folds (default: 10) |
| `-e` | | Number of epochs (default: 10) |
| `-lr` | | Learning rate (default: 0.00001) |
| `-dr` | | Dropout rate (default: 0.1) |
| `-bg` | | Training group |
| `--columns` | `-c` | Data columns to use |
| `--label` | `-l` | Label column |
| `--file` | `-f` | File to read from if `-bg` is not provided |



## Project Structure
```
.
├── main.py                         # Entry point
├── config.py                       # Configuration dataclass
├── experiment_handler.py           # Orchestrates experiments
├── sig_test.py                     # Bootstrap and significance testing
├── data_handling/
│   └── strat_fold.py               # Stratified k-fold cross-validation
├── model/
│   ├── cnn.py                      # CNN model
│   ├── rnn.py                      # Attention-based LSTM RNN model
│   └── roberta.py                  # XLM-RoBERTa model
├── preprocessing/
│   ├── pre_nn.py                   # Preprocessing for CNN and RNN
│   ├── pre_roberta.py              # Preprocessing for RoBERTa
│   └── tokenizers/                 # Custom tokenisers
├── train/
│   ├── train.py                    # General training logic
│   └── train_nn.py                 # CNN/RNN specific training
└── utils/
    ├── path_manager.py             # Path management
    └── visualisation.py            # Confusion matrices, boxplots, F1 plots
```
Note: Data handling and splitting utilities are omitted due to aforementioned privacy status.
