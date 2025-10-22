import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, hidden_state):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        hidden_state: (B, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)       # (B, num_pixels, att_dim)
        att2 = self.decoder_att(hidden_state).unsqueeze(1)  # (B, 1, att_dim)
        att = torch.tanh(att1 + att2)              # (B, num_pixels, att_dim)
        alpha = F.softmax(self.full_att(att), dim=1)  # (B, num_pixels, 1)
        context = (encoder_out * alpha).sum(dim=1)    # (B, encoder_dim)
        return context, alpha