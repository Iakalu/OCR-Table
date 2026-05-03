from __future__ import annotations

from .types import BBox, Cell


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def cell_detection_f1(predicted: list[Cell], target: list[Cell], iou_threshold: float = 0.5) -> dict[str, float]:
    matched_targets: set[int] = set()
    true_positive = 0
    for pred in predicted:
        best_idx = -1
        best_iou = 0.0
        for idx, gold in enumerate(target):
            if idx in matched_targets:
                continue
            score = bbox_iou(pred.bbox, gold.bbox)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            true_positive += 1
            matched_targets.add(best_idx)

    false_positive = len(predicted) - true_positive
    false_negative = len(target) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def exact_text_accuracy(predicted: list[Cell], target: list[Cell]) -> float:
    lookup = {(cell.row, cell.col): cell.text.strip() for cell in target}
    total = 0
    correct = 0
    for cell in predicted:
        key = (cell.row, cell.col)
        if key not in lookup:
            continue
        total += 1
        correct += int(cell.text.strip() == lookup[key])
    return correct / max(1, total)
