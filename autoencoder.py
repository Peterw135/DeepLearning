from dataloader import LeafsnapDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim

class NaiveAutoencoder(nn.Module):
    def __init__(self, num_feats, learning_transform):
        # init
        super().__init__()
        self.num_feats = num_feats
        self.learning_transform = learning_transform

        # architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.num_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_feats),
            nn.Sigmoid(),
        )

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

def train_AE_model(model:nn.Module, dataset:LeafsnapDataset, epochs:int, learning_rate:float):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() #MSE loss is the only appropriate loss for AE

    for e in range(epochs):
        for images, _ in dataset: #note, labels are discarded :D
            optimizer.zero_grad()

            tampered_images = model.learning_transform(images)
            reconstructed_images = model(tampered_images)
            loss = criterion(reconstructed_images, images)
            loss.backward()

            optimizer.step()
        print(f"Epoch [{e+1}/{epochs}], Last Loss: {loss.item():.4f}")