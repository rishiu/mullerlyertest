import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.models import resnet18
import os

class MullerLyerDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        img_path = self.data_dir + sorted(os.listdir(self.data_dir))[idx]

        img = read_image(img_path)
        label = torch.tensor([1.0,0.0]) if img_path.find("L") > 0 else torch.tensor([0.0,1.0])

        return img, label

class SimpleMullerLyerModel(nn.Module):
    def __init__(self):
        super(SimpleMullerLyerModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 1, 7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10609, 2048),
            nn.Linear(2048, 2),
            nn.Softmax()
        )

    def forward(self, x):
        y = self.model(x)
        return y


class ResnetMullerLyerModel(nn.Module):
    def __init__(self):
        super(ResnetMullerLyerModel, self).__init__()
        self.resnet = resnet18(pretrained=True).eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(512, 2)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        self.softmax = nn.Softmax()
        print(self.resnet)

    def forward(self, x):
        y = self.resnet(x)
        y = self.softmax(y)
        return y

# if __name__ == "__main__":
#     complex_model = ResnetMullerLyerModel()

