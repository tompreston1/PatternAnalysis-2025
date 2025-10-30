import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataset import ADNIDataset, scan_folder, get_transforms
from modules import ConvNeXtBinaryOptimized as ConvNeXtBinary

# ------------------------------
# Utility
# ------------------------------
@torch.no_grad()
def evaluate_with_metrics(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device).float()
        logits = model(X)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= threshold).int()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    return acc, prec, rec, f1, auc, np.array(y_prob), np.array(y_pred), np.array(y_true)


def plot_sample_predictions(model, dataset, device, save_path="predictions_grid.png", n=8):
    """
    Plots sample predictions (mix of correct and incorrect) with color-coded titles.
    Green title = correct prediction
    Red title = incorrect prediction
    """
    model.eval()
    probs, preds, trues = [], [], []

    # Run inference on all data once
    for img, label in dataset:
        with torch.no_grad():
            img_t = img.unsqueeze(0).to(device)
            prob = torch.sigmoid(model(img_t)).item()
            pred = int(prob >= 0.5)
        probs.append(prob)
        preds.append(pred)
        trues.append(label)

    # Identify correct and incorrect predictions
    indices_correct = [i for i in range(len(dataset)) if preds[i] == trues[i]]
    indices_incorrect = [i for i in range(len(dataset)) if preds[i] != trues[i]]

    # Sample evenly from both if possible
    n_half = n // 2
    # Randomly sample n indices (correct or incorrect, mixed)
    sampled_indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)


    # Plot samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, idx in zip(axes.flat, sampled_indices):
        img, label = dataset[idx]
        prob, pred = probs[idx], preds[idx]
        correct = pred == label

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        ax.imshow(img_np)
        ax.axis("off")
        color = "green" if correct else "red"
        status = "Correct" if correct else "Wrong"
        ax.set_title(
            f"{status}\nPred: {'AD' if pred else 'NC'} ({prob:.2f}) | True: {'AD' if label else 'NC'}",
            color=color, fontsize=10
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved sample predictions to: {save_path}")



# Main
def main(data_root, ckpt_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_dir = os.path.join(data_root, "test")
    test_samples = scan_folder(test_dir)
    transform_eval = get_transforms(train=False)
    test_ds = ADNIDataset(test_samples, transform=transform_eval)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    model = ConvNeXtBinary().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint: {ckpt_path} (threshold=0.5)")

    # Evaluate
    acc, prec, rec, f1, auc, y_prob, y_pred, y_true = evaluate_with_metrics(model, test_loader, device, threshold=0.5)
    print(f"\n Test Metrics — Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    # Plot predictions inline
    plot_sample_predictions(model, test_ds, device)


# Entry
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.data_root, args.ckpt, args.batch_size)
