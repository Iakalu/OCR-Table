from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


BBox = Tuple[int, int, int, int]


def bbox_area(b: BBox) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(b: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class TableBox:
    bbox: BBox
    score: float = 1.0


@dataclass
class Cell:
    row: int
    col: int
    bbox: BBox

    row_span: int = 1
    col_span: int = 1

    text: str = ""
    score: float = 1.0

    meta: dict[str, Any] = field(default_factory=dict)

    def key(self):
        return (self.row, self.col)


@dataclass
class OCRToken:
    text: str
    bbox: BBox
    score: float = 1.0