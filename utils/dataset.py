from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from vocab import Vocabulary


all_captions = [cap for _, cap in data_pairs]
vocab = Vocabulary(freqthreshold=5)
vocab.buildvocabulary(all_captions)


class CocoDataset(Dataset):
    def __init__(self, data_pairs, vocab, transform=None):
        self.data_pairs = data_pairs
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        imgpath, caption = self.data_pairs[idx]
        image = Image.open(imgpath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        numericalized_caption = [self.vocab.stoi["<SOS>"]] + \
            self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        return image, torch.tensor(numericalized_caption)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = [cap.clone() if isinstance(cap, torch.Tensor) else torch.tensor(cap) for cap in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab.stoi["<PAD>"])
    return images, captions
