from __future__ import annotations

from src.table_ocr_pipeline.utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import shutil
import importlib.util


# CORE CHECK
def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_tesseract_exec() -> str:
    return shutil.which("tesseract") or "NOT FOUND"


# BACKEND STATUS
def check_backends() -> dict:
    status = {
        "paddleocr": module_available("paddleocr"),
        "paddle": module_available("paddle"),
        "pytesseract": module_available("pytesseract"),
        "tesseract_exec": check_tesseract_exec(),
        "transformers": module_available("transformers"),
        "torch": module_available("torch"),
    }

    # Derived states 
    status["paddle_ready"] = status["paddleocr"] and status["paddle"]
    status["tesseract_ready"] = status["pytesseract"] and status["tesseract_exec"] != "NOT FOUND"
    status["trocr_ready"] = status["transformers"] and status["torch"]

    return status


# PRETTY PRINT
def print_status(status: dict):
    print("\n=== OCR BACKEND STATUS ===\n")

    def ok(flag: bool) -> str:
        return "O" if flag else "X"

    print("PaddleOCR:")
    print(f"  paddleocr package : {ok(status['paddleocr'])}")
    print(f"  paddle package    : {ok(status['paddle'])}")
    print(f"  usable            : {ok(status['paddle_ready'])}\n")

    print("Tesseract:")
    print(f"  pytesseract       : {ok(status['pytesseract'])}")
    print(f"  executable        : {status['tesseract_exec']}")
    print(f"  usable            : {ok(status['tesseract_ready'])}\n")

    print("TrOCR (Transformers):")
    print(f"  transformers      : {ok(status['transformers'])}")
    print(f"  torch             : {ok(status['torch'])}")
    print(f"  usable            : {ok(status['trocr_ready'])}\n")


# RECOMMENDATION LOGIC
def print_recommendations(status: dict):
    print("=== RECOMMENDATIONS ===\n")

    if status["paddle_ready"]:
        print("PaddleOCR is READY (best choice for this project).\n")
        return

    print("PaddleOCR NOT ready\n")

    if not status["paddleocr"]:
        print("Install paddleocr:")
        print("   pip install paddleocr")

    if not status["paddle"]:
        print("Install paddlepaddle (CPU):")
        print("   pip install paddlepaddle")

    print()

    if not status["tesseract_ready"]:
        print("Optional fallback (Tesseract):")
        print("   1. Install Tesseract app")
        print("   2. pip install pytesseract\n")

    if not status["trocr_ready"]:
        print("Optional deep OCR (TrOCR):")
        print("   pip install transformers sentencepiece torch\n")

    print("After install, run this script again.\n")


def main():
    status = check_backends()

    print_status(status)
    print_recommendations(status)


if __name__ == "__main__":
    main()