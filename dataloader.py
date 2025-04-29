import os
from typing import Callable

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class LeafsnapDataset(Dataset):
    """
    An implementation of a torch.utils.data.Dataset designed to parse the file structure of the LeafSnap dataset.
    https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset
    """
    def __init__(self, leafsnap_TSV_filepath:str, leafsnap_DB_root_dir:str, use_segmented:bool=False, source:str="both", transform:Callable=lambda x:x):
        """
        Initializer that builds the tools for scraping images.

        Arguments:
            leafsnap_TSV_filepath (str): the (absolute or CWD relative) file path pointing to the .txt file describing the data to be used
                    this must be in the same format as leafsnap-dataset-images.txt, since we hardcode the parsing of this file to the structure they use
            leafsnap_DB_root_dir (str): the (absolute or CWD relative) folder path pointing to the leafsnap database where images are stored
                    this must be in the same format as the LeafSnap dataset, since the directory scraping is hardcoded to the structure they use
            use_segmented (bool): if True, __getitem__ will fetch the segmented version of the leaf image and append it as a 4th channel to the image
            source (str in {'both', 'lab', 'field}): determines which set of images are pulled from the database, since the lab and field images have different formatting
            transform (Callable): any callable transform that ingests a tensor (e.g., a torchvision.v2.Compose), __getitem__ will apply it to the image before returning
        """
        self.root_dir = leafsnap_DB_root_dir
        self.use_segmented = use_segmented
        self.transform = transform

        # Load the image source text file.
        self.image_source = pd.read_csv(leafsnap_TSV_filepath, sep='\t')
        # Make sure the source has the right columns.
        required_columns = set(['image_path', 'species'])
        if source != 'both': required_columns.add('source')
        if use_segmented: required_columns.add('segmented_path')
        for col in required_columns: assert col in self.image_source.columns, f'The text file describing the dataset must have a column titled: {col}'

        # Get the specified images.
        if source != "both":
            self.image_source = self.image_source[self.image_source['source'] == source] # get the columns that come from the specified source ('field' or 'lab')
            self.image_source = self.image_source.reset_index(drop=True) # reset the dataframe indexing

        # Build the label mapping
        species_set = sorted(set(self.image_source['species'].unique()))
        self.label_map = {}
        for i, species in enumerate(species_set):
            self.label_map[species] = i

    def __len__(self):
        """
        The length of the dataset.

        Returns:
            length (int)
        """
        return self.image_source.shape[0]

    def __getitem__(self, i):
        """
        Overloads the general item accessor of Dataset.

        Returns:
            item (torch.Tensor): the image in shape (channels, height, width)
                    if use_segmented is True, then there is an additional 4th channel containing the LeafSnap segmentation
                    Note: height and width WILL NOT be consistent across the dataset
            label (int): the label of the image
                    There are 185 classes, so this is in Z[0, 184]
        """
        # Support for negative indexing
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(f"Index {i} out of range for Dataset of length {len(self)}")

        # Find and open the image
        img_path = os.path.join(self.root_dir, self.image_source['image_path'].iloc[i])
        PIL_image = Image.open(img_path).convert("RGB")
        img_tensor = pil_to_tensor(PIL_image) #(3, H, W)

        # Apply segmentation channel if appropriate
        if self.use_segmented:
            seg_path = os.path.join(self.root_dir, self.image_source['segmented_path'].iloc[i])
            PIL_seg = Image.open(seg_path).convert("L")
            seg_tensor = pil_to_tensor(PIL_seg)
            img_tensor = torch.cat([img_tensor, seg_tensor], dim=0)

        item = self.transform(img_tensor) # pass the image through the transform
        label = self.label_map[self.image_source['species'].iloc[i]] # get the i^th label and convert it to the mapping

        return item, label

class FlaviaDataset(Dataset):
    """
    An implementation of a torch.utils.data.Dataset designed to parse the file structure of the Flavia dataset.
    https://sourceforge.net/projects/flavia/
    """
    def __init__(self, img_path, transform = None):
        self.img_path = img_path
        self.transform = transform
        self.data = []
        self.label = []

        for img in os.listdir(self.img_path):
            img_num = int(img.split(".")[0])
            label = self.get_label(img_num)
            if label is not None:
                self.data.append(img_num)
                self.label.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Support for negative indexing
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(f"Index {i} out of range for Dataset of length {len(self)}")

        img_path = os.path.join(self.img_path, str(self.data[i]) + ".jpg")
        image = Image.open(img_path).convert("RGB")
        img_tensor = pil_to_tensor(image) #(3, H, W)

        return self.transform(image), self.label[i]

    def get_label(self, file_num):
        if 1001 <= file_num <= 1059:
            return "01"
        elif 1060 <= file_num <= 1122:
            return "02"
        elif 1552 <= file_num <= 1616:
            return "03"
        elif 1123 <= file_num <= 1194:
            return "04"
        elif 1195 <= file_num <= 1267:
            return "05"
        elif 1268 <= file_num <= 1323:
            return "06"
        elif 1324 <= file_num <= 1385:
            return "07"
        elif 1386 <= file_num <= 1437:
            return "08"
        elif 1497 <= file_num <= 1551:
            return "09"
        elif 1438 <= file_num <= 1496:
            return "010"
        elif 2001 <= file_num <= 2050:
            return "011"
        elif 2051 <= file_num <= 2113:
            return "012"
        elif 2114 <= file_num <= 2165:
            return "013"
        elif 2166 <= file_num <= 2230:
            return "014"
        elif 2231 <= file_num <= 2290:
            return "015"
        elif 2291 <= file_num <= 2346:
            return "016"
        elif 2347 <= file_num <= 2423:
            return "017"
        elif 2424 <= file_num <= 2485:
            return "018"
        elif 2486 <= file_num <= 2546:
            return "019"
        elif 2547 <= file_num <= 2612:
            return "020"
        elif 2616 <= file_num <= 2675:
            return "021"
        elif 3001 <= file_num <= 3055:
            return "022"
        elif 3056 <= file_num <= 3110: 
            return "023"
        elif 3111 <= file_num <= 3175:
            return "024"
        elif 3176 <= file_num <= 3229:
            return "025"
        elif 3230 <= file_num <= 3281:
            return "026"
        elif 3282 <= file_num <= 3334:
            return "027"
        elif 3335 <= file_num <= 3389:
            return "028"
        elif 3390 <= file_num <= 3446:
            return "029"
        elif 3447 <= file_num <= 3510:
            return "030"
        elif 3511 <= file_num <= 3563:
            return "031"
        elif 3566 <= file_num <= 3621: 
            return "032"
        else:
            return "not reachable but it doesn\'t matter anyways since these labels aren't used"

class CombinedDataset(Dataset):
    """
    Since we're synthesizing several sources, this takes in multiple torch.utils.data.Dataset inheritors and combines their indexing.
    """
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        # Precompute the cumulative sizes to locate items later
        self.cumulative_sizes = []
        cum_sum = 0
        for dataset in self.datasets:
            cum_sum += len(dataset)
            self.cumulative_sizes.append(cum_sum)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0: #support for negative indexing
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for CombinedDataset of length {len(self)}")

        # Find which dataset the idx belongs to
        dataset_idx = self._find_dataset_idx(idx)
        if dataset_idx == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][relative_idx]

    def _find_dataset_idx(self, idx):
        # Binary search
        low, high = 0, len(self.cumulative_sizes) - 1
        while low <= high:
            mid = (low + high) // 2
            if idx < self.cumulative_sizes[mid]:
                if mid == 0 or idx >= self.cumulative_sizes[mid - 1]:
                    return mid
                high = mid - 1
            else:
                low = mid + 1
        raise RuntimeError(f"Index {idx} out of bounds despite checks.")
