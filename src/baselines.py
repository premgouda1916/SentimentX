import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification

import numpy as np

# ---------------------------------------------------------
# 1. Traditional Machine Learning Baselines
# ---------------------------------------------------------
def train_traditional_baselines(texts_train, labels_train, texts_test, labels_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, labels_train)
        preds = model.predict(X_test)
        acc = accuracy_score(labels_test, preds)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
    return results

# ---------------------------------------------------------
# 2. Deep Learning Baseline Architectures (PyTorch)
# ---------------------------------------------------------
class RNNBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=6, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        return self.fc(hidden)

class CNNLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conv_out = torch.relu(self.conv1d(embedded)).permute(0, 2, 1)
        out, (hidden, _) = self.lstm(conv_out)
        return self.fc(hidden[-1])

# ---------------------------------------------------------
# 3. Transformer Baselines (HuggingFace)
# ---------------------------------------------------------
def get_transformer_baseline(model_name, num_classes=6):
    """
    Valid model_name inputs:
    - 'bert-base-multilingual-cased'
    - 'xlm-roberta-base'
    - 't5-small' (requires slightly different training logic as it is seq2seq)
    - 'xlnet-base-cased' (standalone XLNet)
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    return model

if __name__ == "__main__":
    # Simulate data for ML models
    X_train_text = ["happy", "crying sad", "angry hate", "scared fear", "shocked wow", "disgusting eww"]
    y_train = [0, 1, 2, 3, 4, 5]
    X_test_text = ["happy joy", "scared"]
    y_test = [0, 3]
    
    train_traditional_baselines(X_train_text, y_train, X_test_text, y_test)
    
    # Test initialize RNN
    model = LSTMBaseline(vocab_size=1000, embed_dim=128, hidden_dim=64, bidirectional=True)
    out = model(torch.randint(0, 1000, (2, 50)))
    print("BiLSTM Output Shape:", out.shape)
