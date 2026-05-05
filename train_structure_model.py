from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw

# dùng đúng data module
from src.table_ocr_pipeline.data.fintabnet_loader import (
    sample_from_hf_structure_item,
    sample_from_manifest_item,
    stream_hf_dataset,
    stream_jsonl_url,
)

# model đúng
from src.table_ocr_pipeline.model.structure_model import (
    build_line_segmentation_model,
)


# Synthetic data
def create_training_sample(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    rows = random.randint(3, 8)
    cols = random.randint(3, 7)
    margin = random.randint(12, 32)
    line_width = random.randint(1, 3)

    image = Image.new("L", (size, size), 255)
    vertical_mask = Image.new("L", (size, size), 0)
    horizontal_mask = Image.new("L", (size, size), 0)

    draw_img = ImageDraw.Draw(image)
    draw_v = ImageDraw.Draw(vertical_mask)
    draw_h = ImageDraw.Draw(horizontal_mask)

    x_positions = np.linspace(margin, size - margin, cols + 1).astype(int)
    y_positions = np.linspace(margin, size - margin, rows + 1).astype(int)

    # jitter
    x_positions += np.random.randint(-4, 5, size=x_positions.shape)
    y_positions += np.random.randint(-4, 5, size=y_positions.shape)

    x_positions = np.clip(x_positions, 0, size - 1)
    y_positions = np.clip(y_positions, 0, size - 1)

    # draw lines
    for x in x_positions:
        draw_img.line((x, y_positions[0], x, y_positions[-1]), fill=random.randint(0, 40), width=line_width)
        draw_v.line((x, y_positions[0], x, y_positions[-1]), fill=255, width=max(3, line_width + 2))

    for y in y_positions:
        draw_img.line((x_positions[0], y, x_positions[-1], y), fill=random.randint(0, 40), width=line_width)
        draw_h.line((x_positions[0], y, x_positions[-1], y), fill=255, width=max(3, line_width + 2))

    # noise
    draw_noise = ImageDraw.Draw(image)
    for _ in range(random.randint(12, 36)):
        x1 = random.randint(margin, size - margin - 25)
        y1 = random.randint(margin, size - margin - 10)
        x2 = x1 + random.randint(8, 35)
        y2 = y1 + random.randint(2, 5)
        draw_noise.rectangle((x1, y1, x2, y2), fill=random.randint(60, 150))

    arr = np.asarray(image).astype("float32") / 255.0
    noise = np.random.normal(0, random.uniform(0.0, 0.04), arr.shape).astype("float32")
    arr = np.clip(arr + noise, 0, 1)

    mask = np.stack(
        [
            np.asarray(vertical_mask).astype("float32") / 255.0,
            np.asarray(horizontal_mask).astype("float32") / 255.0,
        ],
        axis=0,
    )

    return arr[None, :, :], mask


# Data generator
def build_data_iterator(args) -> Iterable:
    if args.data_source == "manifest-url":
        return itertools.cycle(
            itertools.islice(stream_jsonl_url(args.manifest_url), args.max_remote_samples)
        )

    if args.data_source == "hf-fintabnet-pubtables":
        return itertools.cycle(
            itertools.islice(
                stream_hf_dataset(
                    args.hf_dataset,
                    split=args.hf_split,
                    config_name=args.hf_config,
                ),
                args.max_remote_samples,
            )
        )

    return None


def sample_batch(args, remote_iter):
    images = []
    masks = []

    for _ in range(args.batch_size):
        if args.data_source == "manifest-url":
            img, mask = sample_from_manifest_item(next(remote_iter), args.image_size)

        elif args.data_source == "hf-fintabnet-pubtables":
            img, mask = sample_from_hf_structure_item(next(remote_iter), args.image_size)

        else:
            img, mask = create_training_sample(args.image_size)

        images.append(img)
        masks.append(mask)

    return np.stack(images), np.stack(masks)


# Train
def main() -> None:
    parser = argparse.ArgumentParser(description="Train structure line segmentation model")

    parser.add_argument("--data-source", choices=["synthetic", "manifest-url", "hf-fintabnet-pubtables"], default="synthetic")
    parser.add_argument("--manifest-url", default=None)

    parser.add_argument("--hf-dataset", default="katphlab/fintabnet-pubtables-full")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")

    parser.add_argument("--max-remote-samples", type=int, default=2000)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)

    parser.add_argument("--output", default="checkpoints/structure_line_cnn.pt")

    args = parser.parse_args()

    if args.data_source == "manifest-url" and not args.manifest_url:
        raise ValueError("--manifest-url required")

    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_line_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    remote_iter = build_data_iterator(args)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(args.steps_per_epoch):
            x_np, y_np = sample_batch(args, remote_iter)

            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            y = torch.tensor(y_np, dtype=torch.float32, device=device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        print(f"epoch={epoch:02d} loss={total_loss / args.steps_per_epoch:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "image_size": args.image_size,
        },
        output,
    )

    print(f"saved checkpoint -> {output}")


if __name__ == "__main__":
    main()