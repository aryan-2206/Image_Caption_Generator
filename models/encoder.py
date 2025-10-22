import torch.nn as nn
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout_p=0.3):
        super(EncoderCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.fc = nn.Linear(512, embed_size)

    def forward(self, images):
        # images: (B, 3, H, W)
        features = self.cnn(images)              # (B, 512, H', W')
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, H', W', 512)
        features = features.view(B, H*W, C)      # (B, num_pixels, 512)
        features = self.fc(features)             # (B, num_pixels, embed_size)
        return features