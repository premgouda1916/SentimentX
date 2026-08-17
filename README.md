# SentimentX

A Hybrid CNN-XLNet Framework for Multilingual Low-Resource Indian Language Sentiment Analysis.

## Overview
This project provides an end-to-end framework to analyze sentiment spanning multiple languages: Kannada, Malayalam, Hindi, Marathi, Tulu, Telugu, and English. By combining **CNN** for extracting local features and **XLNet** for deep global contextual embeddings, the hybrid architecture achieves state-of-the-art sentiment multi-classification (Positive, Negative, Neutral) even on low-resource datasets.

## Project Structure
```text
SentimentX/
├── api/                  # FastAPI web server and endpoints
│   └── main.py
├── data/                 # Raw and processed datasets
├── frontend/             # Beautiful web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── src/                  # Core ML modules
│   ├── data_prep.py      # Data loader, normalization, Tokenizer, SMOTE
│   ├── model.py          # PyTorch Hybrid CNN-XLNet model definition
│   ├── train.py          # Training & Optimization loop
│   ├── evaluate.py       # Validation and Metrics
│   ├── baselines.py      # Traditional ML & Deep Learning Baselines
│   └── inference.py      # Single prediction pipeline
├── Dockerfile            # Containerization 
└── requirements.txt      # Python dependencies
```

## Setup Instructions
1. Install Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API Server:
   ```bash
   uvicorn api.main:app --reload
   ```
4. Open the frontend:
   Simply open `frontend/index.html` in any modern web browser or serve it over a local static server.

## Pipeline Details
- **Data Preparation**: Simulates loading 'IndianSentimentMultilingual-2026', handling emoji tokens, back-translation data augmentation representations, and applying `imbalanced-learn` SMOTE to assure balanced class representation.
- **Model**: Passes tokens to `xlnet-base-cased` returning hidden states. Conv1D layers varying in kernel sizes operate over these states to capture N-gram semantics, max-pooled and concatenated, evaluated via linear layers with Softmax.
- **Baselines**: For empirical rigor, `src/baselines.py` exposes comparisons to XGBoost, LSTM, BiLSTM, distinct Transformers (BERT, RoBERTa), and standard CNNs.

## Authors
Expert AI Engineer
