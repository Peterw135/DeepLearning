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