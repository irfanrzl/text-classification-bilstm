# Text Classification with BiLSTM

A deep-learning text classification pipeline comparing LSTM and BiLSTM
architectures, built with TensorFlow/Keras. Covers text preprocessing,
exploratory analysis, and model tuning.

## Pipeline

| Notebook | Description |
|----------|-------------|
| `01_preprocessing.ipynb` | Text cleaning, deduplication, label encoding, tokenization, and train/test split |
| `02_eda.ipynb` | Class distribution, text-length, word-frequency, n-gram, and vocabulary analysis |
| `03_model_building.ipynb` | Baseline LSTM, baseline BiLSTM, and a Keras-Tuner–tuned BiLSTM, with comparison |

## Approach

Raw text is cleaned (lowercasing, punctuation removal, empty/duplicate
removal), labels are encoded, and text is tokenized into padded sequences.
EDA characterises the classes and vocabulary before modelling. Three models
are trained and compared on a held-out test set using accuracy, precision,
recall, and F1.

## Results

| Model | Test Accuracy | Precision | Recall | F1 |
|-------|--------------|-----------|--------|-----|
| Baseline LSTM | 0.633 | 0.690 | 0.633 | 0.622 |
| Baseline BiLSTM | **0.703** | 0.727 | 0.703 | 0.703 |
| Tuned BiLSTM | 0.686 | 0.719 | 0.686 | 0.683 |

The bidirectional LSTM outperformed the plain LSTM, as expected for
sequence tasks where context from both directions helps. Notably, the
**baseline BiLSTM outperformed the tuned version** — the hyperparameter
search did not improve generalisation on the test set, a reminder that
tuning does not guarantee gains and that a simpler configuration can
generalise better.

## Getting Started

```bash
git clone https://github.com/<your-username>/text-classification-bilstm.git
cd text-classification-bilstm
pip install -r requirements.txt
```

Run the notebooks in order (01 → 03). The dataset is not included; the
preprocessing notebook documents the expected columns (`text`, `label`).

## Tech Stack

Python · TensorFlow/Keras · Keras Tuner · scikit-learn · pandas · matplotlib

## Author

Muhammad Irfan Bin Mohd Rizal
