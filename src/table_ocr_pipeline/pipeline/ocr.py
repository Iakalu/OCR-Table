from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from ..utils.types import OCRToken


class OCRReader:

    def __init__(self, config: dict):
        self.config = config

        # CONFIG
        ocr_cfg = config.get("ocr", {})

        self.backend = ocr_cfg.get("backend", "auto")
        self.lang = ocr_cfg.get("lang", "en")
        self.debug_dir = Path(ocr_cfg["debug_dir"]) if ocr_cfg.get("debug_dir") else None

        # BACKENDS
        self._paddle = None
        self._tesseract = None
        self._trocr = None


    # PUBLIC API
    def read(self, image_np) -> str:

        image = Image.fromarray(image_np)

        token = self.recognize_cell(image)
        return token.text

    def recognize_cell(self, image: Image.Image) -> OCRToken:

        image = self._prepare_cell_for_ocr(image)


        # PaddleOCR (priority)
        if self.backend in {"auto", "paddleocr"}:
            paddle = self._load_paddle()
            if paddle:
                return self._recognize_paddle(image, paddle)


        # Tesseract fallback
        if self.backend in {"auto", "tesseract"}:
            tesseract = self._load_tesseract()
            if tesseract:
                return self._recognize_tesseract(image, tesseract)


        # TrOCR fallback
        if self.backend in {"auto", "trocr"}:
            trocr = self._load_trocr()
            if trocr:
                return self._recognize_trocr(image, trocr)

        return OCRToken(text="", bbox=(0, 0, image.width, image.height), score=0.0)

    def recognize_tokens(self, image: Image.Image) -> List[OCRToken]:

        if self.backend in {"auto", "paddleocr"}:
            paddle = self._load_paddle()
            if paddle:
                return self._recognize_paddle_tokens(image.convert("RGB"), paddle)

            self._debug("PaddleOCR unavailable → no tokens")

        return []


    # PREPROCESS
    def _prepare_cell_for_ocr(self, image: Image.Image) -> Image.Image:

        width, height = image.size

        pad_x = max(1, min(6, int(width * 0.04)))
        pad_y = max(1, min(6, int(height * 0.08)))

        if width > pad_x * 2 + 8 and height > pad_y * 2 + 8:
            image = image.crop((pad_x, pad_y, width - pad_x, height - pad_y))

        if image.width < 160 or image.height < 48:
            scale = max(2, min(4, int(180 / max(1, image.width))))
            image = image.resize((image.width * scale, image.height * scale))

        return image.convert("RGB")


    # PADDLE OCR
    def _load_paddle(self):
        if self._paddle is not None:
            return self._paddle

        try:
            from paddleocr import PaddleOCR

            try:
                self._paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    show_log=False
                )
            except Exception:
                self._paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False
                )

            print("✔ PaddleOCR loaded")

        except Exception as exc:
            self._debug(f"PaddleOCR load failed: {type(exc).__name__}: {exc}")
            self._paddle = False

        return self._paddle

    def _recognize_paddle(self, image: Image.Image, ocr) -> OCRToken:
        result = ocr.ocr(np.array(image), cls=True)

        texts = []
        scores = []

        for line in result[0] if result else []:
            if isinstance(line, dict):
                text = line.get("text") or line.get("rec_text") or ""
                score = line.get("score") or line.get("rec_score") or 0.0
            else:
                text = line[1][0]
                score = line[1][1]

            texts.append(str(text))
            scores.append(float(score))

        text = " ".join(texts).strip()
        score = sum(scores) / len(scores) if scores else 0.0

        return OCRToken(
            text=text,
            bbox=(0, 0, image.width, image.height),
            score=score
        )

    def _recognize_paddle_tokens(self, image: Image.Image, ocr) -> List[OCRToken]:
        try:
            result = ocr.ocr(np.array(image), cls=True)
        except Exception as exc:
            self._debug(f"PaddleOCR table OCR failed: {exc}")
            return []

        tokens = []

        for line in result[0] if result else []:
            if isinstance(line, dict):
                text = line.get("text") or line.get("rec_text") or ""
                score = float(line.get("score") or line.get("rec_score") or 0.0)
                box = line.get("box") or line.get("dt_polys") or line.get("points")
            else:
                box = line[0]
                text = line[1][0]
                score = float(line[1][1])

            if not text or box is None:
                continue

            pts = np.asarray(box).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)

            tokens.append(
                OCRToken(
                    text=str(text),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=score
                )
            )

        self._debug(f"PaddleOCR tokens: {len(tokens)}")

        return tokens


    # TESSERACT
    def _load_tesseract(self):
        if self._tesseract is not None:
            return self._tesseract

        try:
            import pytesseract
            self._tesseract = pytesseract
        except Exception:
            self._tesseract = False

        return self._tesseract

    def _recognize_tesseract(self, image: Image.Image, pytesseract) -> OCRToken:
        processed = image.convert("L")

        if processed.width < 120 or processed.height < 40:
            processed = processed.resize((processed.width * 2, processed.height * 2))

        try:
            text = pytesseract.image_to_string(processed, config="--psm 6").strip()
        except Exception:
            self._tesseract = False
            return OCRToken(text="", bbox=(0, 0, image.width, image.height), score=0.0)

        return OCRToken(
            text=text,
            bbox=(0, 0, image.width, image.height),
            score=1.0 if text else 0.0
        )


    # TROCR
    def _load_trocr(self):
        if self._trocr is not None:
            return self._trocr

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            model_name = self.config.get("ocr", {}).get(
                "trocr_model_name",
                "microsoft/trocr-base-printed"
            )

            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            self._trocr = (torch, processor, model, device)

        except Exception:
            self._trocr = False

        return self._trocr

    def _recognize_trocr(self, image: Image.Image, backend) -> OCRToken:
        torch, processor, model, device = backend

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=32)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        return OCRToken(
            text=text,
            bbox=(0, 0, image.width, image.height),
            score=1.0 if text else 0.0
        )

    # DEBUG
    def _debug(self, message: str):
        if not self.debug_dir:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)

        with (self.debug_dir / "ocr_debug.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")