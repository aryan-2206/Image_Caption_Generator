import torch
import torchvision.transforms as transforms
from models import combined_model 
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


checkpoint_path = "/kaggle/input/image-caption-checkpoint/checkpoint_img_caption.pth"

embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

model = CombinedModel(embed_size, hidden_size, vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch, loss = load_checkpoint(model, optimizer, path=checkpoint_path)
print(f"Loaded checkpoint from epoch {epoch+1} with loss {loss:.4f}")