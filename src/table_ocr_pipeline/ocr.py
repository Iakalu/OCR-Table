from __future__ import annotations

from PIL import Image

from .types import OCRToken


class CellOCR:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "auto")
        self.lang = config.get("lang", "en")
        self._paddle = None

    def recognize_cell(self, image: Image.Image) -> OCRToken:
        if self.backend in {"auto", "paddleocr"}:
            paddle = self._load_paddle()
            if paddle:
                return self._recognize_paddle(image, paddle)
        return OCRToken(text="", bbox=(0, 0, image.width, image.height), score=0.0)

    def _load_paddle(self):
        if self._paddle is not None:
            return self._paddle
        try:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        except Exception:
            self._paddle = False
        return self._paddle

    def _recognize_paddle(self, image: Image.Image, ocr) -> OCRToken:
        import numpy as np

        result = ocr.ocr(np.array(image), cls=True)
        texts: list[str] = []
        scores: list[float] = []
        for line in result[0] if result else []:
            texts.append(line[1][0])
            scores.append(float(line[1][1]))
        text = " ".join(texts).strip()
        score = sum(scores) / len(scores) if scores else 0.0
        return OCRToken(text=text, bbox=(0, 0, image.width, image.height), score=score)

