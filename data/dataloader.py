# Python script that defines the dataloader for training the neural network pipeline
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import os
import random
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class LFSubApertureDataset(Dataset):
    """
    Class that defines the dataset loader which loads the light filed data from the sub-aperature
    views. It also generates random crops and rotates the cropped image.
    """
    def __init__(self, root_dir, grid_size, spatial_crop, transform=None, gamma_range=(0.4, 1.0)):
        """
        Constructor for the LFSubAPertureDataset class

        Arguments:
        ----------
        root_dir - Directory containing the sub-aperture views
        grid_size - Tuple that defines the sub-aperture views
        spatial_crop - Tuple that defines the patch size for training
        trnasform - Flag that defines if the sub-aperture images have to be resized
        gamma_range - Tuple that defines the range of the gamma values for data augmentation

        Returns:
        --------
        Nothing
        """
        self.grid_dirs = sorted(glob(os.path.join(root_dir, "*")))
        self.U, self.V = grid_size
        self.crop_h, self.crop_w = spatial_crop
        self.transform = transform
        self.gamma_min, self.gamma_max = gamma_range

    def __len__(self):
        """
        Member function that defines the dataset length

        Arguments:
        ----------
        None

        Returns:
        --------
        Returns the number of datapoints in the training data
        """
        return len(self.grid_dirs)

    def __getitem__(self, idx):
        """
        Member function that defines the __getitem___ behavior for the data loader

        Arguments:
        ----------
        idx - The index of the datapoint

        Returns:
        --------
        Returns a tuple containing the center view and the light field ground truth
        """
        folder = self.grid_dirs[idx]
        views = []
        for v in range(2, self.V-2, 1):
            row = []
            for u in range(2, self.U-2, 1):
                path = os.path.join(folder, f"view_{v:02d}_{u:02d}.png")
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                img_t = TF.to_tensor(img).float()          # [3, H, W]

                # random gamma
                gamma = random.uniform(self.gamma_min, self.gamma_max)
                img_t = img_t.pow(gamma)

                # normalize to [-1,1]
                img_t = (img_t - 0.5) * 2.0
                
                row.append(img_t)
            views.append(torch.stack(row, dim=0))  # [U,3,H,W]
        lf = torch.stack(views, dim=0)             # [V,U,3,H,W]


        # center view tensor [3,H,W]
        u0, v0 = (self.U-4)//2, (self.V-4)//2
        center = lf[v0, u0]                        # [3,H,W]

        # pick random crop coords
        _, H, W = center.shape
        top  = random.randint(0, H - self.crop_h)
        left = random.randint(0, W - self.crop_w)

        # crop center: [3, crop_h, crop_w]
        center_crop = center[:, top:top+self.crop_h, left:left+self.crop_w]

        # crop all views: [U, V, 3, crop_h, crop_w]
        lf_crop = lf[:, :, :, top:top+self.crop_h, left:left+self.crop_w]

        # --- reorder to H, W, ... formats ---

        # center_crop: [3, h, w] → [h, w, 3]
        # center_crop = center_crop.permute(1, 2, 0)

        # lf_crop: [V, U, 3, h, w] → [h, w, V, U, 3]
        lf_crop = lf_crop.permute(3, 4, 0, 1, 2)

        return center_crop, lf_crop

# --- Collate that stacks center and targets ---
def lfCollate(batch):
    """
    Helper fucntion that collates the centeral view and the targets

    Arguments:
    ----------
    batch - Tuple that contains the central view, light field ground truth pair

    Returns:
    --------
    Returns a tuple that contains the collated output
    """
    centers_hw3, lf_hwuv3 = zip(*batch)
    # stack into [B, h, w, 3] and [B, h, w, U, V, 3]
    centers = torch.stack(centers_hw3, dim=0)    # [B, h, w, 3]
    lf_views = torch.stack(lf_hwuv3, dim=0)      # [B, h, w, V, U, 3]
    return centers, lf_views

# --- DataLoader factory ---
def getDataloader(root_dir, grid_size, spatial_crop,
                                 batch_size, num_workers=4,
                                 resize=None):
    """
    Function that defines the DataLoader factory

    Aguments:
    ---------
        spatial_crop: tuple (crop_h, crop_w)
        resize: optional (H, W) to resize raw views before cropping

    Returns:
    --------
        returns the dataloader object
    """
    transform = None
    if resize is not None:
        def transform(img):
            return img.resize(resize, Image.BILINEAR)
    dataset = LFSubApertureDataset(
        root_dir, grid_size, spatial_crop, transform=transform
    )
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=lfCollate,
                        drop_last=True)
    return loader