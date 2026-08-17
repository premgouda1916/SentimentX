import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score, confusion_matrix
import math

from model import HybridCNNXLNet
from data_prep import get_dataloaders, create_synthetic_dataset

def plot_confusion_matrix(y_true, y_pred, classes, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def evaluate_model(model, dataloader, device, classes=['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']):
    model.eval()
    y_true = []
    y_pred = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Standard Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # F2-Score targets cases where Recall is more important
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    
    # Perplexity (Exponential of Cross Entropy Loss)
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
    
    # Note on Generative Metrics (BLEU, ROUGE):
    # These metrics are designed for sequence-to-sequence tasks (machine translation, summarization).
    # Since this architecture (CNN-XLNet) is exclusively a classification model (N -> 1 task),
    # ROUGE and BLEU are strictly not applicable logically. 
    # Returning placeholders to satisfy baseline reporting formatting.
    bleu_score = "N/A (Classification Task)"
    rouge_score = "N/A (Classification Task)"
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'perplexity': perplexity,
        'bleu_score': bleu_score,
        'rouge_score': rouge_score
    }
    
    print(f"Validation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    plot_confusion_matrix(y_true, y_pred, classes=classes)
    
    return metrics

if __name__ == "__main__":
    import os
    from data_prep import load_real_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading real dataset for evaluation...")
    df = load_real_dataset('data')
    
    # Sub-sample dataset on CPU to ensure evaluation runs fast
    if device == "cpu":
        print("Note: Running on CPU. Sub-sampling dataset to 200 rows for fast evaluation.")
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
        
    _, val_loader, tokenizer, _ = get_dataloaders(df, batch_size=16)
    
    model = HybridCNNXLNet(num_classes=6)
    model_path = 'saved_models/sentimentx_best_model.pth'
    
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No trained model found! Evaluating initialized model.")
        
    model.to(device)
    evaluate_model(model, val_loader, device)
