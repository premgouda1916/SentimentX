# SentimentX: A Hybrid CNN-XLNet Framework for Multilingual Low-Resource Indian Language Sentiment Analysis

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Deep Learning Framework](https://img.shields.io/badge/framework-PyTorch_2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![API Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Inference Latency](https://img.shields.io/badge/GPU_Latency-%3C30ms-green.svg)](#performance-metrics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SentimentX** is an end-to-end sentiment analysis framework optimized for low-resource regional Indian languages and phonetic code-mixed scripts. Rather than classifying text into simple positive/negative binary markers, the platform maps inputs to **Ekman's six basic emotions**: *Happiness, Sadness, Anger, Fear, Surprise, and Disgust*.

Under the hood, the framework leverages a hybrid deep learning model combining **XLNet Transformers** (for global contextual semantics) with a **Multi-Kernel 1D Convolutional Neural Network (Conv1D)** (to extract local n-gram semantic patterns and regional slang expressions).

---

## 🌟 Key Features

* **Multilingual Coverage:** Native script support for 5 regional languages (**Kannada, Malayalam, Hindi, Marathi, Telugu**) and **English**.
* **Phonetic Code-Mixing Handling:** Robust handling of transliterated texts (e.g., Kannada/Malayalam written in Latin script characters) utilizing a customized **SentencePiece** subword vocabulary.
* **Granular Emotion Classifier:** 6-class emotional categorization based on Ekman's psychological schema.
* **Real-time Asynchronous Pipeline:** An asynchronous **FastAPI** backend with Uvicorn server providing CPU predictions in under 150ms and CUDA GPU predictions in under 30ms.
* **Premium Dashboard UI:** A beautiful glassmorphic dark-themed web dashboard with live emotion probability charts and browser `LocalStorage` persistent history query logs.

---

## 🛠️ System Architecture

The SentimentX system flows through 4 processing layers:

```text
[ Raw Multilingual Text ]
          │
          ▼
┌─────────────────────────────────┐
│ 1. Linguistic Preprocessing     │ Normalization, Noise Filtering,
│    & Emoji Demojization         │ Emoji to Token Conversion (e.g. 😡 -> :angry_face:)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 2. XLNet Transformer Encoder    │ SentencePiece tokenization,
│    (768-dimensional states)     │ Permutation-based global embeddings
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 3. Parallel Multi-Kernel Conv1D │ Kernels of sizes 3, 4, 5 capture
│    (Feature Extractors)         │ local n-gram constructs (trigrams to 5-grams)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 4. Global Max-Pooling           │ Saliency extraction, Dropout (0.5), Fully Connected
│    & Softmax Classification     │ layers with Softmax mapping to 6 emotion classes
└─────────────────────────────────┘
          │
          ▼
[ 6-Class Emotion Probability Output ]
```

---

## 📂 Repository Structure

```text
SentimentX/
├── api/                        # API backend server
│   └── main.py                 # FastAPI endpoints and predictor setup
├── data/                       # Datasets
│   ├── dataset.csv             # Unified dataset (72,365 records)
│   └── [Lang].csv              # Individual language flat files
├── data_pipeline_scripts/      # Data collection, cleaning, and preprocessing tools
├── frontend/                   # Web interface dashboard
│   ├── index.html              # Glassmorphic user interface structure
│   ├── style.css               # Design style tokens
│   └── script.js               # Client-side persistent cache and API Fetch logic
├── src/                        # Core deep learning modules
│   ├── model.py                # HybridCNNXLNet model definition
│   ├── data_prep.py            # SentencePiece loader & balanced loss weight calculation
│   ├── train.py                # Training loop with AMP, Early Stopping & PowerShell Alert hook
│   ├── evaluate.py             # Model performance evaluator
│   ├── baselines.py            # Baseline comparisons (XGBoost, LSTMs, VADER)
│   └── inference.py            # High-speed predictor orchestrator
├── SentimentX_IEEE_Paper.tex   # LaTeX source for the academic paper
├── Dockerfile                  # Containerization specification
└── requirements.txt            # Python dependencies
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.12+
* CUDA-enabled GPU (optional, but recommended for sub-30ms execution)

### 2. Clone and Setup Environment
Navigate to the directory and set up a virtual environment:
```bash
cd SentimentX
python -m venv venv
venv\Scripts\activate   # For Windows PowerShell/CMD
pip install -r requirements.txt
```

### 3. Run the Backend API
Startup the asynchronous FastAPI server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation will be available locally at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Open the Web Dashboard
Simply launch the frontend in a browser:
```bash
explorer frontend/index.html
```

---

## 📈 Performance & Evaluation

SentimentX was trained and validated on a curated dataset of **72,365 entries**, demonstrating superior performance against standalone Transformers and traditional ML baselines:

| Metric | SentimentX Performance |
| :--- | :--- |
| **Validation Accuracy** | **95.8%** |
| **Precision** | **95.1%** |
| **F1-Score** | **95.5%** |
| **Inference Latency (GPU)** | **< 30 ms** |
| **Inference Latency (CPU)** | **< 150 ms** |

### Training Optimizations Used:
* **Balanced Class Weights:** Loss function penalizations are mathematically scaled using dataset frequencies to counter low-resource minority class data scarcity.
* **Automatic Mixed Precision (AMP):** Utilizing Float16 gradients through `torch.amp.autocast` to reduce GPU memory footprint and accelerate training.
* **Early Stopping:** Configured with a patience of 3 epochs monitoring validation loss to prevent overfitting.

---

## 📄 Academic Citation

If you use this work, please cite our academic paper:
```latex
@inproceedings{gouda2026sentimentx,
  title={SentimentX: A Hybrid Convolutional Neural Network and XLNet Transformer Framework for Multilingual Low-Resource Indian Language Sentiment Analysis},
  author={Gouda, Prem and N, Nikith and Katti, Naveen and T P, Sumanth and Prajapati, Manish},
  booktitle={Proceedings of Yenepoya Department of Information Science & Engineering},
  year={2026}
}
```
