import pandas as pd
import numpy as np
import re
import emoji
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class TextPreprocessor:
    def __init__(self):
        # Extremely basic stopword list for multiple languages (simulated)
        self.stop_words = {"the", "a", "an", "is", "for", "hai", "and", "ki", "en", "njan"}
        
    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def handle_emojis(self, text):
        # Convert emojis to text representation (e.g. :smile:)
        return emoji.demojize(text)

    def remove_stopwords(self, text):
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return " ".join(words)

    def process(self, text):
        text = self.normalize_text(text)
        text = self.handle_emojis(text)
        text = self.remove_stopwords(text)
        return text

class DataAugmenter:
    """ Simulate data augmentation (synonym replacement/back translation) """
    def __init__(self):
        self.synonyms = {
            "good": ["excellent", "great", "nice"],
            "bad": ["terrible", "poor", "awful"],
            "happy": ["joyful", "glad"],
            "sad": ["unhappy", "depressed"]
        }

    def synonym_replacement(self, text, n=1):
        words = text.split()
        if len(words) == 0:
            return text
        new_words = words.copy()
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if word in self.synonyms:
                new_words[idx] = random.choice(self.synonyms[word])
        return " ".join(new_words)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_dataloaders(df, text_col='text', label_col='label', max_len=128, batch_size=32):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', local_files_only=True)
    
    # Text preprocessing
    preprocessor = TextPreprocessor()
    augmenter = DataAugmenter()
    
    # Optional augmentation for training data
    df['cleaned_text'] = df[text_col].apply(preprocessor.process)
    
    texts = df['cleaned_text'].tolist()
    labels = df[label_col].tolist()
    
    # Tokenize (returns token ids and attention masks)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
    
    # Skip SMOTE to avoid memory crash
    y_resampled = labels
    resampled_encodings = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    }
    
    # Compute class weights for imbalanced datasets
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float)
    
    # Train test split
    X_train_ids, X_val_ids, y_train, y_val, mask_train, mask_val = train_test_split(
        resampled_encodings['input_ids'], y_resampled, resampled_encodings['attention_mask'], test_size=0.2, random_state=42
    )

    train_encodings = {'input_ids': X_train_ids, 'attention_mask': mask_train}
    val_encodings = {'input_ids': X_val_ids, 'attention_mask': mask_val}

    train_dataset = SentimentDataset(train_encodings, y_train)
    val_dataset = SentimentDataset(val_encodings, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer, class_weights_tensor

def load_real_dataset(path="data"):
    import os
    import glob
    
    emotion_map_reverse = {
        'Happiness': 0,
        'Sadness': 1,
        'Anger': 2,
        'Fear': 3,
        'Surprise': 4,
        'Disgust': 5
    }
    
    resolved_path = path
    # If path is not found or has no CSV files, check under SentimentX/path (e.g. SentimentX/data)
    if not os.path.exists(resolved_path) or (os.path.isdir(resolved_path) and not glob.glob(os.path.join(resolved_path, "*.csv"))):
        fallback_path = os.path.join("SentimentX", path)
        if os.path.exists(fallback_path) and glob.glob(os.path.join(fallback_path, "*.csv")):
            resolved_path = fallback_path
            print(f"Note: Redirecting dataset search to fallback path '{resolved_path}'")

    if os.path.isfile(resolved_path):
        csv_files = [resolved_path]
    elif os.path.isdir(resolved_path):
        csv_files = glob.glob(os.path.join(resolved_path, "*.csv"))
        # Force deprecation of massive monolithic file
        csv_files = [f for f in csv_files if os.path.basename(f) != "dataset.csv"]
    else:
        print(f"Warning: Path '{resolved_path}' not found. Falling back to synthetic.")
        return create_synthetic_dataset(100)
    
    if not csv_files:
        print(f"Warning: No valid CSVs found in '{resolved_path}'. Falling back to synthetic.")
        return create_synthetic_dataset(100)
        
    dfs = []
    for file in csv_files:
        try:
            lang_df = pd.read_csv(file)
            if 'label' in lang_df.columns and lang_df['label'].dtype == object:
                lang_df['label'] = lang_df['label'].map(emotion_map_reverse)
            dfs.append(lang_df)
        except Exception:
            pass
            
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df = unified_df.dropna(subset=['label'])
    unified_df['label'] = unified_df['label'].astype(int)
    return unified_df

def create_synthetic_dataset(num_samples=1000):
    texts = [
        "This product is amazing and I love it!", # Happiness
        "It's a terrifying experience.",          # Fear
        "I am so incredibly angry right now 😡", # Anger
        "valare nallathu, loved it very much",    # Happiness (Malayalam code-mixed)
        "ನನಗೆ ತುಂಬಾ ದುಃಖವಾಗಿದೆ",                   # Sadness (Kannada: I am very sad)
        "Wow! I never expected this to happen!",  # Surprise
        "This food smells absolutely foul.",      # Disgust
    ]
    labels = [0, 3, 2, 0, 1, 4, 5] 
    # 0=Happiness, 1=Sadness, 2=Anger, 3=Fear, 4=Surprise, 5=Disgust
    
    syn_texts = []
    syn_labels = []
    for _ in range(num_samples):
        idx = random.randint(0, len(texts) - 1)
        syn_texts.append(texts[idx])
        syn_labels.append(labels[idx])
        
    df = pd.DataFrame({'text': syn_texts, 'label': syn_labels})
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    df = load_real_dataset('data')
    train_loader, val_loader, tokenizer = get_dataloaders(df, batch_size=8)
    print("Data Loaders created successfully.")
    for batch in train_loader:
        print(batch['input_ids'].shape)
        break
