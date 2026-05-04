from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import List

from ..utils.types import Cell  


# CELLS → MATRIX
def cells_to_matrix(cells: List[Cell]) -> List[List[str]]:

    if not cells:
        return []

    max_row = max(cell.row + max(0, cell.row_span - 1) for cell in cells)
    max_col = max(cell.col + max(0, cell.col_span - 1) for cell in cells)

    matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in cells:
        r, c = cell.row, cell.col

        row_span = max(1, cell.row_span)
        col_span = max(1, cell.col_span)

        matrix[r][c] = cell.text

        # fill spanned area
        for dr in range(row_span):
            for dc in range(col_span):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if rr < len(matrix) and cc < len(matrix[0]):
                    matrix[rr][cc] = ""
    return matrix



# CSV
def write_csv(matrix: List[List[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)



# HTML
def write_html(matrix: List[List[str]], path: str | Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []

    for row_idx, row in enumerate(matrix):
        tag = "th" if row_idx == 0 else "td"

        cells_html = "".join(
            f"<{tag}>{escape(value)}</{tag}>"
            for value in row
        )

        rows.append(f"<tr>{cells_html}</tr>")

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



# JSON
def write_json(cells: List[Cell], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "row": cell.row,
            "col": cell.col,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "bbox": list(cell.bbox),  # CHANGED: ensure JSON serializable
            "text": cell.text,
            "score": float(cell.score),
        }
        for cell in cells
    ]

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )