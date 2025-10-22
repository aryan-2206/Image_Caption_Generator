import torch
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import os
import json
from models import combined_model
from utils import dataset 

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint_img_caption.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, dataloader, criterion, optimizer, vocab, device, num_epochs=1):
    model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)
        
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]
        
            # Handle possible mismatch
            min_len = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]
        
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)
        
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, avg_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='blue')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return train_losses

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths and prepare dataset
datasetpath = "/kaggle/input/coco-2017-dataset/coco2017"
trainimagepath = os.path.join(datasetpath, "train2017")
annotationspath = os.path.join(datasetpath, "annotations")
captionsjsonpath = os.path.join(annotationspath, "captions_train2017.json")

with open(captionsjsonpath, "r") as f:
    captionsjson = json.load(f)

idtofilename = {img["id"]: img["file_name"] for img in captionsjson["images"]}

data_pairs = []
for ann in captionsjson["annotations"]:
    img_id = ann["image_id"]
    if img_id in idtofilename:
        data_pairs.append((os.path.join(trainimagepath, idtofilename[img_id]), ann["caption"]))

all_captions = [cap for _, cap in data_pairs]
vocab = Vocabulary(freqthreshold=5)
vocab.buildvocabulary(all_captions)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = CocoDataset(data_pairs, vocab, transform=transform)
subset_dataset = Subset(dataset, list(range(60000)))

dataloader = DataLoader(
    subset_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

model = CombinedModel(embed_size, hidden_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
train_model(model, dataloader, criterion, optimizer, vocab, device, num_epochs=60)