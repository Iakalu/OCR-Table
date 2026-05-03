from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

from .types import Cell


def cells_to_matrix(cells: list[Cell]) -> list[list[str]]:
    if not cells:
        return []
    max_row = max(cell.row for cell in cells)
    max_col = max(cell.col for cell in cells)
    matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        matrix[cell.row][cell.col] = cell.text
    return matrix


def write_csv(matrix: list[list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(matrix)


def write_html(matrix: list[list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for row_idx, row in enumerate(matrix):
        tag = "th" if row_idx == 0 else "td"
        cells = "".join(f"<{tag}>{escape(value)}</{tag}>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Table OCR Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #222; padding: 8px 14px; }}
    th {{ background: #eef4ff; }}
  </style>
</head>
<body>
  <table>
    {"".join(rows)}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_json(cells: list[Cell], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "row": cell.row,
            "col": cell.col,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "bbox": cell.bbox,
            "text": cell.text,
            "score": cell.score,
        }
        for cell in cells
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

