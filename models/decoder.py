import torch
import torch.nn as nn
from attention import Attention

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, dropout=0.3):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_out, captions):
        """
        encoder_out: (B, num_pixels, embed_size)
        captions: (B, T)
        """
        batch_size = encoder_out.size(0)
        embeddings = self.embedding(captions[:, :-1])  # (B, T-1, embed)
        h = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_out.device)
        c = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_out.device)
        outputs = []

        for t in range(embeddings.size(1)):
            context, _ = self.attention(encoder_out, h)  # (B, embed_size)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)  # (B, 2*embed)
            h, c = self.lstm(lstm_input, (h, c))
            out = self.fc(self.dropout(h))  # (B, vocab_size)
            outputs.append(out.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)  # (B, T-1, vocab_size)
        return outputs