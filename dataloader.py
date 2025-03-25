import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LeafsnapDataset(Dataset):
    def __init__(self, image_path, root_folder, use_segmented=False, source="both", transform=None):
        self.root_directory = root_folder
        self.transform = transform

        self.image_info = pd.read_csv(image_path, sep='\t')

        if source != "both":
            self.image_info = self.image_info[self.image_info['source'] == source]
            self.image_info = self.image_info.reset_index(drop=True)

        if use_segmented:
            self.img_path_col = 'segmented_path'
        else:
            self.img_path_col = 'image_path'

        label_hashmap = {}
        species_set = set(sorted(self.image_info['species'].unique()))

        for i, species in enumerate(species_set):
            label_hashmap[species] = i

        self.labels = []
        for species in self.image_info['species']:
            self.labels.append(label_hashmap[species])

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, i):
        img_path = os.path.join(self.root_directory, self.image_info[self.img_path_col][i])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), self.labels[i]