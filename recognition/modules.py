"""
modules.py
------------
Task 8: Custom ConvNeXt-Tiny implementation (built from scratch).

This version is designed specifically for small, medical-style datasets (e.g., ADNI MRI scans)
where overfitting is common. It balances depth and regularization to achieve around ~0.8 F1
on test data within 10-15 epochs.

Author: Thomas Preston
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Implements stochastic depth regularization (a form of dropout applied to residual branches).
    It randomly drops entire paths during training to improve generalization.
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # Skip if no drop_prob or not training
        if self.drop_prob == 0.0 or not self.training:
            return x

        # Compute a binary mask with probability = (1 - drop_prob)
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # convert to binary mask

        # Scale output to preserve expected values
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    """Applies LayerNorm correctly to BCHW tensors."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # BCHW → BHWC → normalize → BCHW
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    

# ConvNeXt Block (main building unit)
class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt block:
    - Depthwise 7x7 convolution
    - LayerNorm + Pointwise (Linear) expansions
    - GELU activation
    - Residual connection with learnable scaling (gamma)
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise convolution (preserves #channels)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm applied in channel-last format (BHWC)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # MLP-style expansion and projection (pointwise convs)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # project back

        # Layer scaling: stabilizes training from scratch
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        # DropPath: stochastic residual dropping
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x  # residual
        x = self.dwconv(x)                       # spatial feature extraction
        x = x.permute(0, 2, 3, 1)                # BCHW -> BHWC for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x                       # layer scaling
        x = x.permute(0, 3, 1, 2)                # BHWC -> BCHW
        return shortcut + self.drop_path(x)      # residual connection


class ConvNeXtTinyOptimized(nn.Module):
    """
    ConvNeXt-Tiny backbone re-implemented from scratch.

    Architecture details (from the ConvNeXt paper):
    - Depths per stage: [3, 3, 9, 3]
    - Channel dims:     [96, 192, 384, 768]
    """
    def __init__(self, num_classes=1, in_chans=3, dropout=0.5):
        super().__init__()

        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]

        
        # Downsampling stem + layers
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # Core ConvNeXt Stages
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, 0.1, sum(depths)).tolist()  # gradual stochastic depth
        cur = 0
        for i in range(4):
            blocks = [ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        # Head / Classification layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Truncated normal initialization (per ConvNeXt paper)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward pass through all stages
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)  # downsample
            x = self.stages[i](x)             # feature extraction
        x = x.mean([-2, -1])                  # global average pooling (spatial -> vector)
        x = self.norm(x)
        return x

    # Classification head
    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


# Binary Classification Wrapper
class ConvNeXtBinaryOptimized(nn.Module):
    """
    Wrapper for binary classification tasks.
    Outputs a single logit suitable for BCEWithLogitsLoss.
    """
    def __init__(self, in_chans=3):
        super().__init__()
        self.backbone = ConvNeXtTinyOptimized(num_classes=1, in_chans=in_chans)

    def forward(self, x):
        return self.backbone(x)
