import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split

from dataset import ADNIDataset, scan_folder, get_transforms
from modules import ConvNeXtBinaryOptimized as ConvNeXtBinary


# Utility: Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Training and evaluation
def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.05):
    model.train()
    losses, y_true, y_pred = [], [], []

    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device).float().unsqueeze(1)
        optimizer.zero_grad()

        logits = model(X)
        smoothed_y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        loss = criterion(logits, smoothed_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy().flatten()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for X, y in tqdm(loader, desc=desc, leave=False):
        X, y = X.to(device), y.to(device).float().unsqueeze(1)
        logits = model(X)
        loss = criterion(logits, y)
        losses.append(loss.item())

        preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy().flatten()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc


# Main training logic
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset ---
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")

    train_samples = scan_folder(train_dir)
    test_samples = scan_folder(test_dir)

    if not train_samples or not test_samples:
        raise ValueError("Dataset folders missing or empty.")

    # --- Stratified split (preserves AD/NC ratio) ---
    labels = [label for _, label in train_samples]
    train_samples, val_samples = train_test_split(
        train_samples, test_size=0.15, stratify=labels, random_state=args.seed
    )

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print("Train distribution:", Counter(l for _, l in train_samples))
    print("Val distribution:", Counter(l for _, l in val_samples))
    print("Test distribution:", Counter(l for _, l in test_samples))

    # --- Datasets and loaders ---
    transform_train = get_transforms(train=True)
    transform_eval = get_transforms(train=False)

    train_ds = ADNIDataset(train_samples, transform=transform_train)
    val_ds = ADNIDataset(val_samples, transform=transform_eval)
    test_ds = ADNIDataset(test_samples, transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model setup ---
    model = ConvNeXtBinary(dropout=0.3).to(device)  # add dropout for regularization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine Annealing with Warmup
    def warmup_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    best_val_acc, patience, max_patience = 0, 0, 8

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), args.ckpt)
            print("Model improved — checkpoint saved.")
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping triggered.")
                break

    # --- Final test ---
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\nFinal Test Results — Loss={test_loss:.4f}, Acc={test_acc:.3f}")
    print("Training complete.")


# Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)   
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--ckpt', default='checkpoints/best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
