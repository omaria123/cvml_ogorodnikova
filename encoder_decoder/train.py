import os
import random
import string
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
from torchvision import transforms
from torch import nn
import torch.optim as optim

class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, mode=1):
        super().__init__()
        self.n = n 
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.n
    
    def _get_random_text(self, length):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if self.mode == 1:
            text = "ABC"
            x = random.randint(10, self.size - 60)
            y = random.randint(10, self.size - 40)
            
        elif self.mode == 2:
            text = self._get_random_text(3)
            x, y = 30, 30
            
        elif self.mode == 3:
            length = random.randint(1, 8)
            text = self._get_random_text(length)
            x, y = 30, 30
            
        elif self.mode == 4:
            length = random.randint(1, 8)
            text = self._get_random_text(length)
            x = random.randint(10, self.size - 60)
            y = random.randint(10, self.size - 40)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor 


class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x 


def train_mode(mode_id, epochs=5):
    encoder = Encoder()
    decoder = Decoder()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(n=2000, size=256, mode=mode_id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    encoder.to(device)
    decoder.to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            latent = encoder(imgs)
            output = decoder(latent)
            
            loss = criterion(imgs, output)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

    torch.save(encoder.state_dict(), f"encoder_mode_{mode_id}.pth")
    torch.save(decoder.state_dict(), f"decoder_mode_{mode_id}.pth")
    print(f"Модели для режима {mode_id} сохранены")


if __name__ == "__main__":
    for mode in range(1, 5):
        train_mode(mode, epochs=10) 