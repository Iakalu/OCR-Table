# Cai dat OCR de doc noi dung trong cell

Neu output CSV/HTML/JSON da co cau truc bang nhung cac o rong, nghia la structure model da chay nhung OCR backend chua hoat dong.

## Kiem tra OCR backend

```powershell
python check_ocr_backend.py
```

Ban can it nhat mot backend OCR hoat dong.

## Cach khuyen dung: PaddleOCR

```powershell
pip install paddleocr paddlepaddle
```

Sau do chay:

```powershell
python check_ocr_backend.py
streamlit run app.py
```

Trong web chon `Lightweight` hoac `Full pretrained`.

## Cach fallback: Tesseract

`pytesseract` chi la wrapper Python. Ban phai cai ca app Tesseract OCR cua Windows.

1. Tai Tesseract Windows: https://github.com/UB-Mannheim/tesseract/wiki
2. Cai vao:

```text
C:\Program Files\Tesseract-OCR
```

3. Them vao PATH:

```text
C:\Program Files\Tesseract-OCR
```

4. Mo terminal moi:

```powershell
tesseract --version
pip install pytesseract
python check_ocr_backend.py
```

## Cach deep OCR optional: TrOCR

```powershell
pip install transformers sentencepiece torch torchvision
```

Lan dau chay se tai model `microsoft/trocr-base-printed`.

## Luu y quan trong

- Structure model chi dung de tach hang/cot/cell.
- OCR backend moi la phan doc chu trong cell.
- Neu khong co PaddleOCR/Tesseract/TrOCR, output co the co cell nhung `text` se rong.
