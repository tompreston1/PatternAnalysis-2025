# predict.py — Evaluate saved ConvNeXtBinary model on ADNI test set
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADNIDataset, scan_folder, get_transforms
from modules import ConvNeXtBinaryOptimized as ConvNeXtBinary


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_prob = [], []

    for X, y in tqdm(loader, desc="Testing", leave=False):
        X = X.to(device)
        y = y.float().unsqueeze(1).to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        y_prob.extend(probs)
        y_true.extend(y.cpu().numpy().flatten())

    preds = (np.array(y_prob) > threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    print(f"\n✅ Test Metrics — Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    ConfusionMatrixDisplay(cm, display_labels=["NC (0)", "AD (1)"]).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return np.array(y_true), np.array(preds), np.array(y_prob)


def show_samples(dataset, model, device, threshold=0.5, n_samples=8):
    """Display random predictions from test set."""
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples // 2, figsize=(16, 6))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        img, label = dataset[idx]
        X = img.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(X)).item()
        pred = int(prob > threshold)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{label}  P:{pred}  ({prob:.2f})",
                     color="green" if label == pred else "red")

    plt.tight_layout()
    plt.show()


def main(data_root="ADNI/AD_NC", ckpt_path="checkpoints/best_model.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load test data ---
    test_dir = os.path.join(data_root, "test")
    test_samples = scan_folder(test_dir)
    transform = get_transforms(train=False)
    test_ds = ADNIDataset(test_samples, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load model + threshold ---
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ConvNeXtBinary(dropout=0.3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    threshold = ckpt.get("threshold", 0.5)
    print(f"Loaded model checkpoint: {ckpt_path} (threshold={threshold:.2f})")

    # --- Evaluate ---
    y_true, y_pred, y_prob = evaluate(model, test_loader, device, threshold=threshold)

    # --- Show some predictions ---
    show_samples(test_ds, model, device, threshold=threshold, n_samples=8)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.data_root, args.ckpt, args.batch_size)
