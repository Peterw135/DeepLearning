from typing import Tuple, Callable

from dataloader import LeafsnapDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim

class AEClassifier(nn.Module):
    def __init__(self, encoder:nn.Module, embedded_dim:int, num_classes:int):
        super().__init__()
        self.encoder = encoder # Use the trained encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedded_dim, 300),
            nn.ReLU(),
            nn.Linear(300, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass input through encoder
        x = self.classifier(x)  # Pass through classification head
        return x

class _Autoencoder(nn.Module):
    def __init__(self, input_shape:Tuple, learning_transform:Callable): # call this
        super().__init__()

        assert isinstance(input_shape, tuple), 'Input shape must be a tuple.'
        assert len(input_shape) == 4, 'Input shape must be (B, C, H, W).'

        self.input_shape = input_shape
        self.learning_transform = learning_transform

    def embed(self, x): # override this
        raise NotImplementedError

class NaiveAutoencoder(_Autoencoder):
    def __init__(self, input_shape:Tuple, embedded_dim:int, learning_transform:Callable=lambda x: x):
        # init
        super().__init__(input_shape, learning_transform)
        self.vector_dim = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
        self.embedded_dim = embedded_dim

        # architecture
        self.encoder = nn.Sequential(
            nn.Flatten(), # squash (C, H, W)
            nn.Linear(self.vector_dim, self.embedded_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedded_dim, self.vector_dim),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=self.input_shape[1:]), # unsquash (C, H, W)
        )

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

def train_AE_model(model:_Autoencoder, dataset:LeafsnapDataset, epochs:int, learning_rate:float):
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

def train_classifier_head(model:nn.Module, dataset:LeafsnapDataset, epochs:int, learning_rate:float):
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        for images, labels in dataset: #note, labels are discarded :D

            optimizer.zero_grad()

            estimated_labels = model(images)
            loss = criterion(estimated_labels, labels)
            loss.backward()

            optimizer.step()
        print(f"Epoch [{e+1}/{epochs}], Last Loss: {loss.item():.4f}")
