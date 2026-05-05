# Cai dat, chay demo, va tuning

## 1. Cai dat nhanh

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Chay web:

```powershell
streamlit run app.py
```

Chay CLI:

```powershell
python demo.py --output-dir outputs/demo
```

## 2. OCR backend (neu CSV/HTML/JSON bi rong text)

Kiem tra backend:

```powershell
python check_ocr_backend.py
```

### PaddleOCR (khuyen dung)

```powershell
pip install -U paddleocr paddlepaddle
```

Ghi chu Windows:
- Neu PaddleOCR bi loi runtime (PIR/oneDNN), pipeline se tu fallback sang backend khac (Tesseract/TrOCR) neu co.

### Tesseract (fallback nhe)

`pytesseract` chi la wrapper Python, can cai them Tesseract app tren Windows (UB Mannheim build), sau do:

```powershell
pip install -U pytesseract
tesseract --version
```

### TrOCR (fallback deep OCR)

```powershell
pip install -U transformers sentencepiece torch torchvision
```

## 3. Tuning nhanh theo tinh huong

### Miss table (khong detect ra bang)
- Giam `detection.confidence_threshold`
- Tang `preprocess.max_side`

### Sai hang/cot (structure lech)
- Dieu chinh `structure.merge_tolerance`
- Dieu chinh `structure.min_cell_width/min_cell_height`

### OCR sai / mat chu (chu nho, o mau)
- Tang `ocr.cell_padding`
- Tune `ocr.preprocess` (contrast/sharpness/binarize)

## 4. Where to tune trong repo?

- `configs/full_pipeline.yaml`: full pretrained (Table Transformer + OCR).
- `configs/default.yaml`: auto/fallback, nhe hon.

