import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel

class HybridCNNXLNet(nn.Module):
    def __init__(self, pretrained_model_name='xlnet-base-cased', num_classes=6, cnn_filters=128, dropout_rate=0.5):
        super(HybridCNNXLNet, self).__init__()
        
        # 1. XLNet (for global contextual understanding)
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_name, local_files_only=True)
        
        # Hidden size for xlnet-base-cased is 768
        xlnet_hidden_size = self.xlnet.config.d_model
        
        # 2. CNN Layers (for local feature extraction)
        # We will use multiple kernel sizes to capture different n-gram features
        self.conv_layer_3 = nn.Conv1d(in_channels=xlnet_hidden_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv_layer_4 = nn.Conv1d(in_channels=xlnet_hidden_size, out_channels=cnn_filters, kernel_size=4, padding=2)
        self.conv_layer_5 = nn.Conv1d(in_channels=xlnet_hidden_size, out_channels=cnn_filters, kernel_size=5, padding=2)
        
        # 3. Fully Connected Layers
        # Concatenated size depends on the number of filters and kernel sizes
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(cnn_filters * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Step 1: XLNet Contextual Embeddings
        xlnet_output = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        
        # We take the sequence of hidden states from XLNet: (batch_size, seq_len, hidden_size)
        sequence_output = xlnet_output.last_hidden_state
        
        # Reshape for Conv1D: (batch_size, channels, seq_len)
        # channels == hidden_size
        sequence_output = sequence_output.permute(0, 2, 1)
        
        # Step 2: CNN Layers applied to sequence
        # (batch_size, cnn_filters, seq_len')
        conv3 = F.relu(self.conv_layer_3(sequence_output))
        conv4 = F.relu(self.conv_layer_4(sequence_output))
        conv5 = F.relu(self.conv_layer_5(sequence_output))
        
        # Step 3: Max Pooling over time (seq_len)
        # Pool across the entire sequence dimension
        pooled3 = F.max_pool1d(conv3, kernel_size=conv3.shape[2]).squeeze(2)
        pooled4 = F.max_pool1d(conv4, kernel_size=conv4.shape[2]).squeeze(2)
        pooled5 = F.max_pool1d(conv5, kernel_size=conv5.shape[2]).squeeze(2)
        
        # Step 4: Concatenate Features
        # Output shape: (batch_size, cnn_filters * 3)
        cat_features = torch.cat((pooled3, pooled4, pooled5), dim=1)
        
        # Step 5: Fully Connected and Output
        x = self.dropout(cat_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

if __name__ == '__main__':
    # A simple self-test
    model = HybridCNNXLNet()
    dummy_input_ids = torch.randint(0, 1000, (4, 128))
    dummy_attn_mask = torch.ones((4, 128))
    output = model(dummy_input_ids, dummy_attn_mask)
    print("Model Output Shape:", output.shape) # Expected: (4, 6)
