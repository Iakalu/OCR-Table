from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.table_ocr_pipeline.utils.config import load_config
from src.table_ocr_pipeline.evaluation.evaluate_structure import cell_detection_f1, exact_text_accuracy
from src.table_ocr_pipeline.pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.utils.types import Cell


def load_cells(path: str | Path) -> list[Cell]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    cells: list[Cell] = []

    for item in payload:
        cells.append(
            Cell(
                row=int(item["row"]),
                col=int(item["col"]),
                bbox=tuple(int(v) for v in item["bbox"]),
                row_span=int(item.get("row_span", 1)),
                col_span=int(item.get("col_span", 1)),
                text=item.get("text", ""),
                score=float(item.get("score", 1.0)),
            )
        )

    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Table OCR Pipeline")

    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--target-json", required=True, help="Ground truth JSON")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    args = parser.parse_args()

    image_path = Path(args.image)
    target_json = Path(args.target_json)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not target_json.exists():
        raise FileNotFoundError(f"Target JSON not found: {target_json}")

    print(f"\nImage        : {image_path}")
    print(f"Ground truth : {target_json}")
    print(f"Config       : {args.config}")

    print("\nRunning pipeline...\n")

    config = load_config(args.config)
    pipeline = TableOCRPipeline(config)

    result = pipeline.run(image_path, args.output_dir)

    predicted = load_cells(result["json"])
    target = load_cells(target_json)

    metrics = cell_detection_f1(
        predicted,
        target,
        iou_threshold=args.iou_threshold,
    )

    metrics["text_accuracy"] = exact_text_accuracy(predicted, target)

    print("\n=== EVALUATION RESULT ===\n")

    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"Text Acc  : {metrics['text_accuracy']:.4f}")

    print("\nDetails:")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")

    out_path = Path(args.output_dir) / "metrics.json"

    out_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()