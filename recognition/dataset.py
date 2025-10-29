import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
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
                # ✅ Label only based on final directory name
                folder = os.path.basename(os.path.dirname(path)).lower()
                if folder == 'ad':
                    label = 1
                elif folder in ('nc', 'cn'):
                    label = 0
                else:
                    continue
                samples.append((path, label))
    return samples
