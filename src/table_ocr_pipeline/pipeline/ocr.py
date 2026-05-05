from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
                try:
                    return self._recognize_paddle(image, paddle)
                except NotImplementedError as exc:
                    # Some Paddle/PaddleX builds on Windows hit PIR/oneDNN runtime issues.
                    # Disable Paddle backend and continue with fallbacks.
                    self._debug(f"PaddleOCR runtime failed: {type(exc).__name__}: {exc}")
                    self._paddle = False
                except Exception as exc:
                    self._debug(f"PaddleOCR runtime failed: {type(exc).__name__}: {exc}")
                    self._paddle = False


        # Tesseract fallback
        if self.backend in {"auto", "tesseract"}:
            tesseract = self._load_tesseract()
            if tesseract:
                return self._recognize_tesseract(image, tesseract)


        # TrOCR fallback
        enable_trocr_fallback = bool((self.config.get("ocr", {}) or {}).get("enable_trocr_fallback", False))
        if self.backend == "trocr" or (self.backend == "auto" and enable_trocr_fallback):
            trocr = self._load_trocr()
            if trocr:
                return self._recognize_trocr(image, trocr)

        return OCRToken(text="", bbox=(0, 0, image.width, image.height), score=0.0)

    def recognize_tokens(self, image: Image.Image) -> List[OCRToken]:

        if self.backend in {"auto", "paddleocr"}:
            paddle = self._load_paddle()
            if paddle:
                try:
                    return self._recognize_paddle_tokens(image.convert("RGB"), paddle)
                except NotImplementedError as exc:
                    self._debug(f"PaddleOCR token runtime failed: {type(exc).__name__}: {exc}")
                    self._paddle = False
                except Exception as exc:
                    self._debug(f"PaddleOCR token runtime failed: {type(exc).__name__}: {exc}")
                    self._paddle = False

            self._debug("PaddleOCR unavailable â†’ no tokens")

        return []


    # PREPROCESS
    def _prepare_cell_for_ocr(self, image: Image.Image) -> Image.Image:
        cfg = (self.config.get("ocr", {}) or {}).get("preprocess", {}) or {}

        width, height = image.size

        pad_x = max(1, min(6, int(width * 0.04)))
        pad_y = max(1, min(6, int(height * 0.08)))

        if width > pad_x * 2 + 8 and height > pad_y * 2 + 8:
            image = image.crop((pad_x, pad_y, width - pad_x, height - pad_y))

        if image.width < 160 or image.height < 48:
            scale = max(2, min(4, int(180 / max(1, image.width))))
            image = image.resize((image.width * scale, image.height * scale))

        image = image.convert("RGB")

        # extra clarity for small text on colored backgrounds
        if bool(cfg.get("denoise", True)):
            image = image.filter(ImageFilter.MedianFilter(size=3))

        image = ImageOps.autocontrast(image, cutoff=int(cfg.get("autocontrast_cutoff", 1)))
        image = ImageEnhance.Contrast(image).enhance(float(cfg.get("contrast", 1.35)))
        image = ImageEnhance.Sharpness(image).enhance(float(cfg.get("sharpness", 1.6)))

        # Optional binarize for time strings / OFF / PH
        if bool(cfg.get("binarize", True)):
            gray = image.convert("L")
            arr = np.asarray(gray).astype("float32")
            # percentile threshold works well for screenshots
            p = float(cfg.get("binarize_percentile", 55))
            thr = float(np.percentile(arr, p))
            bw = (arr > thr).astype("uint8") * 255
            image = Image.fromarray(bw, mode="L").convert("RGB")

        return image

    # PADDLE OCR
    def _load_paddle(self):
        if self._paddle is not None:
            return self._paddle

        try:
            # Reduce Windows runtime issues (PIR/oneDNN) seen in some paddle builds.
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_enable_onednn", "0")

            from paddleocr import PaddleOCR

            def _try_create(lang: str):
                # PaddleOCR arguments vary across versions. Try conservative constructors first
                # after the legacy show_log form fails.
                attempts = [
                    {"use_angle_cls": True, "lang": lang, "show_log": False},
                    {"use_angle_cls": True, "lang": lang},
                    {"lang": lang},
                    {"use_angle_cls": True, "lang": "en"},
                    {"lang": "en"},
                    {},
                ]
                last_error = None
                for kwargs in attempts:
                    try:
                        return PaddleOCR(**kwargs)
                    except Exception as exc:
                        last_error = exc
                raise last_error if last_error else RuntimeError("Unable to initialize PaddleOCR")

            try:
                self._paddle = _try_create(self.lang)
            except Exception:
                self._paddle = _try_create("en")

            print("[ok] PaddleOCR loaded")

        except Exception as exc:
            self._debug(f"PaddleOCR load failed: {type(exc).__name__}: {exc}")
            self._paddle = False

        return self._paddle

    def _normalize_paddle_lines(self, result):

        if result is None:
            return []
        if isinstance(result, list):
            if not result:
                return []
            first = result[0]
            if first is None:
                return []
            if isinstance(first, list):
                return [ln for ln in first if ln is not None]
            return [ln for ln in result if ln is not None]
        return []
    
    def _recognize_paddle(self, image: Image.Image, ocr) -> OCRToken:
        img = np.array(image)
        try:
            result = ocr.ocr(img, cls=True)
        except TypeError as exc:
            # PaddleOCR API differences across versions:
            # some versions route kwargs into predict() which may not accept `cls`.
            if "cls" in str(exc) and ("unexpected" in str(exc).lower() or "got an unexpected keyword argument" in str(exc).lower()):
                result = ocr.ocr(img)
            else:
                raise
        except NotImplementedError:
            # Bubble up to caller to fallback.
            raise

        texts = []
        scores = []

        for line in self._normalize_paddle_lines(result):
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
            img = np.array(image)
            try:
                result = ocr.ocr(img, cls=True)
            except TypeError as exc:
                if "cls" in str(exc) and (
                    "unexpected" in str(exc).lower() or "got an unexpected keyword argument" in str(exc).lower()
                ):
                    result = ocr.ocr(img)
                else:
                    raise
            except NotImplementedError:
                raise
        except Exception as exc:
            self._debug(f"PaddleOCR table OCR failed: {exc}")
            return []

        tokens = []

        for line in self._normalize_paddle_lines(result):
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
