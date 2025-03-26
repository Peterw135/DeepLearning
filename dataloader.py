import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LeafsnapDataset(Dataset):
    def __init__(self, image_path, root_folder, use_segmented=False, source="both", transform=None):
        self.root_directory = root_folder
        self.transform = transform

        self.data = []
        with open(image_path, 'r') as f:
            col_names = f.readline().strip().split('\t')

            for line in f:
                row = line.strip().split('\t')
                self.data.append(row)

        self.data = np.array(self.data)

        if source != "both":
            source_col = col_names.index('source')
            self.data = self.data[self.data[:, source_col] == source]

        if use_segmented:
            self.img_path_col = col_names.index('image_path')
            self.segmented_path_col = col_names.index('segmented_path')
        else:
            self.img_path_col = col_names.index('image_path')
            self.segmented_path_col = None

        label_hashmap = {}
        species_col = col_names.index('species')

        species_set = sorted(set(self.data[:, species_col]))

        for i, species in enumerate(species_set):
            label_hashmap[species] = i

        self.labels = []
        for species in self.data[:, species_col]:
            self.labels.append(label_hashmap[species])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path = os.path.join(self.root_directory, self.data[i, self.img_path_col])
        image = Image.open(img_path).convert("RGB")

        if self.segmented_path_col is not None:
            segmented_path = os.path.join(self.root_directory, self.data[i, self.segmented_path_col])
            segmented_image = Image.open(segmented_path).convert("L")
            segmented_image = segmented_image.resize(image.size)
            segmented_image = np.array(segmented_image)
            image = np.array(image)
            image = np.dstack((image, segmented_image))
            image = Image.fromarray(image)

        return self.transform(image), self.labels[i]