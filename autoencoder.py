from typing import Tuple, Callable, List
from torch.utils.data import DataLoader

from dataloader import LeafsnapDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class AEClassifier(nn.Module):
    def __init__(self, encoder:nn.Module, layer_sizes:List[int], device):
        super().__init__()
        self.encoder = encoder.to(device) # Use the trained encoder
        self.device = device

        linear_layers = []
        for i in range(len(layer_sizes) - 1):
            linear_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            linear_layers.append(nn.ReLU())
        linear_layers.pop(-1)
        linear_layers.append(nn.Softmax(1))
        self.classifier = nn.Sequential(*linear_layers).to(device)

    def forward(self, x):
        x.to(self.device) # make sure the Tensor is on the device
        encoded = self.encoder(x)  # Pass input through encoder
        logits = self.classifier(encoded)  # Pass through classification head
        return logits

class _Autoencoder(nn.Module):
    def __init__(self, input_shape:Tuple, learning_transform:Callable, device): # call this
        super().__init__()

        assert isinstance(input_shape, tuple), 'Input shape must be a tuple.'
        assert len(input_shape) == 4, 'Input shape must be (B, C, H, W).'

        self.input_shape = input_shape
        self.learning_transform = learning_transform
        self.device = device

        self.to(device)

    def embed(self, x): # override this
        raise NotImplementedError

class ConvolutionalAutoencoder(_Autoencoder):
    def __init__(self, input_shape:Tuple, learning_transform:Callable, device):
        # init
        super().__init__(input_shape, learning_transform, device)
        assert self.input_shape[1:]==(3, 400, 400), 'Image must be 3ch by 400px by 400px'

        # (3, 400, 400)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(25)
        )
        # (25, 400, 400)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (25, 200, 200)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(50)
        )
        # (50, 200, 200)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (50, 100, 100)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=75, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(75)
        )
        # (75, 100, 100)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (75, 50, 50)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=100, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(100)
        )
        # (100, 50, 50)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (100, 25, 25)
        self.enter_dense = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=8, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(8)
        )
        # (8, 25, 25)
        self.flattener = nn.Flatten()
        # (5000,)

        # (5000,)
        self.recovery = nn.Unflatten(dim=1, unflattened_size=(8, 25, 25))
        # (8, 25, 25)
        self.exit_dense = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=100, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(100)
        )
        # (100, 25, 25)
        self.up4 = nn.Sequential( # Note: using convolution to make this a learnable upsampling
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(100)
        )
        # (100, 50, 50)
        self.unconv4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=75, kernel_size=3, padding='same', padding_mode='replicate'), # Note: not using "deconvolution"
            nn.BatchNorm2d(75)
        )
        # (75, 50, 50)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=75, out_channels=75, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(75)
        )
        # (75, 100, 100)
        self.unconv3 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=50, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(50)
        )
        # (50, 100, 100)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(50)
        )
        # (50, 200, 200)
        self.unconv2 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(25)
        )
        # (25, 200, 200)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(25)
        )
        # (25, 400, 400)
        self.unconv1 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=3, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.Sigmoid()
        )
        # (3, 400, 400)

        self.encoder = nn.Sequential(self.conv1, self.down1, self.conv2, self.down2, self.conv3, self.down3, self.conv4, self.down4, self.enter_dense, self.flattener)
        self.decoder = nn.Sequential(self.recovery, self.exit_dense, self.up4, self.unconv4, self.up3, self.unconv3, self.up2, self.unconv2, self.up1, self.unconv1)

        self.to(device)

    def embed(self, x):
        x.to(self.device) # make sure the Tensor is on the device
        return self.encoder(x)

    def forward(self, x):
        x.to(self.device) # make sure the Tensor is on the device
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed




def train_AE_model(model:_Autoencoder, dataloader:DataLoader, epochs:int, learning_rate:float, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() #MSE loss is the only appropriate loss for AE

    for e in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f'Epoch {e+1}/{epochs}')
        for batch_idx, (images, _) in enumerate(progress_bar): #note, labels are discarded, bc AE :D
            images = images.to(device)

            optimizer.zero_grad()
            tampered_images = model.learning_transform(images).to(device)
            reconstructed_images = model(tampered_images)
            loss = criterion(reconstructed_images, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), batch=f'{batch_idx+1}/{num_batches}')
        print(f'Total epoch loss: {epoch_loss}')

def train_classifier_head(model:nn.Module, dataloader:DataLoader, epochs:int, learning_rate:float, device):
    model.to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f'Epoch {e+1}/{epochs}')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            estimated_labels = model(images)
            loss = criterion(estimated_labels, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), batch=f'{batch_idx+1}/{num_batches}')
        print(f'Total epoch loss: {epoch_loss}')
