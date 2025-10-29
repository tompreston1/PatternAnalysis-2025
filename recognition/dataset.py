import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image   
from torchvision import transforms


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

      # Convert to RGB if needed
      if self.rgb:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

      # ✅ Convert NumPy → PIL (torchvision expects PIL)
      img = Image.fromarray(img)

      # Apply torchvision transforms
      if self.transform:
          img = self.transform(img)

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


def get_transforms(train=True):
    """
    Optimized torchvision transforms for MRI classification (ConvNeXt-compatible).
    - Converts grayscale MRI -> RGB (3 channels)
    - Adds subtle augmentations for robustness without distorting features
    - Uses normalization centered at 0.5 for ConvNeXt pretraining alignment
    """

    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.15, saturation=0.05, hue=0.02
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
