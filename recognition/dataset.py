import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ADNIDataset(Dataset):
    """
    Loads individual MRI slices as images for Alzheimer’s vs Control classification.
    - Supports grayscale or RGB input
    - Applies percentile clipping + normalization for consistency
    """
    def __init__(self, samples, transform=None, rgb=True):
        self.samples = samples
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Percentile clipping (reduces brightness outliers)
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        # Convert to RGB if required
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply transformations
        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


def scan_folder(base_dir):
    """
    Scans AD/NC subfolders for image files and labels them correctly.
    Labels:
        AD (Alzheimer's Disease) -> 1
        NC/CN (Normal Control) -> 0
    """
    samples = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                # Label only based on final directory name
                folder = os.path.basename(os.path.dirname(path)).lower()
                if folder == 'ad':
                    label = 1
                elif folder in ('nc', 'cn'):
                    label = 0
                else:
                    continue
                samples.append((path, label))
    return samples

def get_transforms(train=True, size=224, rgb=True):
    """
    Returns simple, fast Albumentations transforms.
    - Light augmentations (flip, brightness/contrast, small rotation)
    - Normalization compatible with ConvNeXtTinyOptimized
    """
    mean = (0.5, 0.5, 0.5) if rgb else (0.5,)
    std = (0.5, 0.5, 0.5) if rgb else (0.5,)

    if train:
        return A.Compose([
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.08), rotate=(-15, 15), shear=(-5, 5), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.4),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
])

    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


