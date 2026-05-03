from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.table_ocr_pipeline.config import load_config
from src.table_ocr_pipeline.evaluation import cell_detection_f1, exact_text_accuracy
from src.table_ocr_pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.types import Cell


def load_cells(path: str | Path) -> list[Cell]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
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
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against a JSON cell annotation.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = TableOCRPipeline(load_config(args.config)).run(args.image, args.output_dir)
    predicted = load_cells(result["json"])
    target = load_cells(args.target_json)
    metrics = cell_detection_f1(predicted, target, iou_threshold=args.iou_threshold)
    metrics["text_accuracy"] = exact_text_accuracy(predicted, target)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
