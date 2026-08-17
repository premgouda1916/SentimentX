import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import XLNetTokenizer
import torch.nn.functional as F
from src.model import HybridCNNXLNet
from src.data_prep import TextPreprocessor

class SentimentPredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', local_files_only=True)
        self.preprocessor = TextPreprocessor()
        self.labels = {0: 'Happiness', 1: 'Sadness', 2: 'Anger', 3: 'Fear', 4: 'Surprise', 5: 'Disgust'}
        
        self.model = HybridCNNXLNet(num_classes=6)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: Initialized untrained model weights.")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        clean_text = self.preprocessor.process(text)
        encoding = self.tokenizer(clean_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        label = self.labels[predicted_class.item()]
        score = confidence.item()
        
        return {
            "text": text,
            "label": label,
            "confidence": score
        }

if __name__ == "__main__":
    predictor = SentimentPredictor(model_path='saved_models/sentimentx_best_model.pth')
    test_texts = [
        "The newly launched scheme logic is terrible, nothing works! 😡",
        "It's an okay movie.",
        "valare nallathu, loved it very much!"  # Mixed code setup
    ]
    for text in test_texts:
        res = predictor.predict(text)
        # Avoid console encoding issues with emojis
        encoding = sys.stdout.encoding or 'utf-8'
        safe_text = res['text'].encode(encoding, errors='replace').decode(encoding)
        print(f"Input: {safe_text}")
        print(f"Predicted Sentiment: {res['label']} (Confidence: {res['confidence']:.4f})\n")
