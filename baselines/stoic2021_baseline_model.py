''' Baseline Model for STOIC2021 - Logistic Regression Model '''

# Standard Library
import os


## External Libraries
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from medpy.io import load
import csv
import matplotlib.pyplot as plt

data_dir = os.path.abspath("trimmed")
train_file = os.path.abspath("./metadata/train.csv")
val_file = os.path.abspath("./metadata/val.csv")
test_file = os.path.abspath("./metadata/test.csv")

# Dataset Class for DataLoader
img_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class STOIC2021Dataset(Dataset):
    def __init__(self, data_dir, annotations_file, transforms=None):
        self.data_dir = data_dir # where the CT scan images are
        self.transforms = transforms
        with open(annotations_file, newline="") as f:
            reader = csv.reader(f)
            self.image_labels = list(reader)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_labels[idx][0] + ".mha")
        image, _= load(img_path)
        label = self.image_labels[idx][1]

        if self.transforms:
            image = self.transforms(image)
        return image, label

val_dataset = STOIC2021Dataset(data_dir=data_dir, annotations_file=val_file, transforms=img_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for val_features, val_labels in val_dataloader:
    print(f"Feature batch shape: {val_features.size()}")
    print(f"Labels batch shape: {len(val_labels)}")
    img = val_features[0].squeeze()
    label = val_labels[0]
    print(f"Label: {label}")
