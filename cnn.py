from typing import Tuple, Callable, List
from torch.utils.data import DataLoader

from dataloader import LeafsnapDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from tqdm import tqdm


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features, device):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, device, kernel, stride=1):
        super(BasicBlock, self).__init__()
        self.device = device
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x.to(self.device) 

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_layers, num_classes, device):
        super(ResNet_s, self).__init__()
        self.device = device
        self.in_planes = 32

        self.conv1 = nn.Conv2d(num_layers, self.in_planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], device=device, kernel=5, stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], device=device, kernel=5, stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], device=device, kernel=5, stride=2)

        self.linear = nn.Linear(128, num_classes)
        self.apply(_weights_init)

        self.num_layers = num_layers

    def _make_layer(self, block, planes, num_blocks, device, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, device, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x.to(self.device) 

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(num_layers, num_classes, device):
    return ResNet_s(BasicBlock, [5, 5, 5], num_layers, num_classes=num_classes, device=device)

def train_resnet_model(model:ResNet_s, dataloader:DataLoader, epochs:int, learning_rate:float, device):
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  for epoch in range(epochs):
    running_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
    for i, (images, labels) in enumerate(progress_bar, 0):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), batch=f'{i+1}/{num_batches}')
    print(f'Epoch {epoch} loss: {running_loss / len(dataloader)}')
