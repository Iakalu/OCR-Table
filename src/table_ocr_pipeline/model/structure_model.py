# src/models/structure_model.py

from __future__ import annotations

import torch
import torch.nn as nn


# Model Definition
class LineSegNet(nn.Module):

    def __init__(self):
        super().__init__()


        # Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 2, kernel_size=1),  # 2 channels: horiz + vert
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Factory 
def build_structure_model() -> LineSegNet:
    """
    NEW: Thay thế build_line_segmentation_model()
    """
    return LineSegNet()


# Load Model (robust)
def load_structure_model(checkpoint_path: str, device: str = "cpu") -> LineSegNet:

    model = LineSegNet().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # assume raw state_dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# Save Model (chuẩn hóa training)
def save_structure_model(model: nn.Module, path: str):
    torch.save(
        {
            "model_state": model.state_dict()
        },
        path
    )