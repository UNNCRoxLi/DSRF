import os
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


# sub_folders = 'None'


class RPV(Dataset):
    def __init__(
        self, dataset_dir, data_split=None, image_size=80, 
        transform=None, subset=None
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform

        # subsets = os.listdir(self.dataset_dir)

        self.file_names = []
        # for i in subsets:
        file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, "*_" + self.data_split + ".npz"))]
        self.file_names = file_names


    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]

        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        image = data["image"].reshape(16, 360, 640, 3)


        if self.image_size != 160:
            resize_image = np.zeros((16, 3, self.image_size, self.image_size))
            for idx in range(0, 16):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size), 
                    interpolation = cv2.INTER_NEAREST
                ).transpose(2,0,1)
        else:
            resize_image = image

        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self._get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = torch.tensor(0)
        structure = torch.tensor(0)
        structure_encoded = torch.tensor(0)
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)         
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)

        return image, target, meta_target, structure_encoded, data_file

    