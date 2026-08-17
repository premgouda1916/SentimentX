import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import subprocess

def notify_user(title, message):
    """Sends a Windows notification using PowerShell."""
    try:
        powershell_command = f"""
        [reflection.assembly]::loadwithpartialname('System.Windows.Forms');
        [System.Windows.Forms.MessageBox]::Show('{message}', '{title}')
        """
        subprocess.run(["powershell", "-Command", powershell_command], capture_output=True)
    except Exception as e:
        print(f"Notification failed: {e}")


from data_prep import get_dataloaders, create_synthetic_dataset, load_real_dataset
from model import HybridCNNXLNet

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, class_weights, epochs=5, lr=2e-5, device='cpu'):
    model.to(device)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    early_stopping = EarlyStopping(patience=3)
    
    # Enable mixed precision for faster training if on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float('inf')
    os.makedirs('saved_models', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).long()
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).long()
                
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                total_val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 30)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join('saved_models', 'sentimentx_best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")
            
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            msg = f"Early stopping triggered at Epoch {epoch+1}. Model has reached maximum efficiency!"
            print(msg)
            notify_user("SentimentX Training", msg)
            break
            
    if not early_stopping.early_stop:
        msg = f"Training completed all {epochs} epochs successfully!"
        print(msg)
        notify_user("SentimentX Training", msg)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data from the data folder (falls back to synthetic if missing)
    os.makedirs('data', exist_ok=True)
    df = load_real_dataset('data')
    
    # Sub-sample dataset on CPU to ensure training runs fast
    if device == "cpu":
        print("Note: Running on CPU. Sub-sampling dataset to 1000 rows for fast execution.")
        df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    
    train_loader, val_loader, tokenizer, class_weights = get_dataloaders(df, batch_size=16)
    
    # 2. Init Model
    model = HybridCNNXLNet(num_classes=6)
    
    # 3. Load checkpoint if available (resume from saved weights)
    checkpoint_path = os.path.join('saved_models', 'sentimentx_best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} — resuming from saved weights...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint found. Starting from pre-trained XLNet weights.")
    
    # 4. Train
    # To run a real training sweep, set epochs > 1 and load the real dataset
    train_model(model, train_loader, val_loader, class_weights, epochs=5, device=device)
