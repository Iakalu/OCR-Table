from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[int, int, int, int]


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


@dataclass
class OCRToken:
    text: str
    bbox: BBox
    score: float = 1.0

