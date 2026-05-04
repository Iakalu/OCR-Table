from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..utils.types import BBox, Cell


# IOU
def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# MATCHING
def match_cells(
    predicted: List[Cell],
    target: List[Cell],
    iou_threshold: float
) -> List[Tuple[int, int, float]]:

    if not predicted or not target:
        return []

    cost_matrix = np.zeros((len(predicted), len(target)), dtype=np.float32)

    for i, p in enumerate(predicted):
        for j, t in enumerate(target):
            cost_matrix[i, j] = bbox_iou(p.bbox, t.bbox)

    # Greedy fallback if scipy not available
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            iou = cost_matrix[i, j]
            if iou >= iou_threshold:
                matches.append((i, j, iou))

        return matches

    except Exception:
        # fallback greedy
        matches = []
        used_targets = set()

        for i in range(len(predicted)):
            best_j = -1
            best_iou = 0.0

            for j in range(len(target)):
                if j in used_targets:
                    continue

                iou = cost_matrix[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                matches.append((i, best_j, best_iou))
                used_targets.add(best_j)

        return matches


# STRUCTURE METRIC
def cell_detection_f1(
    predicted: List[Cell],
    target: List[Cell],
    iou_threshold: float = 0.5
) -> dict[str, float]:

    matches = match_cells(predicted, target, iou_threshold)

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(target) - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


# TEXT METRIC
def exact_text_accuracy(predicted: List[Cell], target: List[Cell]) -> float:

    lookup = {
        (c.row, c.col): c.text.strip()
        for c in target
    }

    total = 0
    correct = 0

    for cell in predicted:
        key = (cell.row, cell.col)

        if key not in lookup:
            continue

        total += 1

        if cell.text.strip() == lookup[key]:
            correct += 1

    return correct / max(1, total)



# TEXT ACCURACY WITH MATCHING
def text_accuracy_with_iou(
    predicted: List[Cell],
    target: List[Cell],
    iou_threshold: float = 0.5
) -> float:

    matches = match_cells(predicted, target, iou_threshold)

    if not matches:
        return 0.0

    correct = 0

    for pi, ti, _ in matches:
        p = predicted[pi]
        t = target[ti]

        if p.text.strip() == t.text.strip():
            correct += 1

    return correct / len(matches)



# NEW: FULL REPORT
def evaluate_table(
    predicted: List[Cell],
    target: List[Cell],
    iou_threshold: float = 0.5
) -> dict:

    structure = cell_detection_f1(predicted, target, iou_threshold)

    text_exact = exact_text_accuracy(predicted, target)
    text_iou = text_accuracy_with_iou(predicted, target, iou_threshold)

    return {
        "structure": structure,
        "text_exact": text_exact,
        "text_iou": text_iou,
    }