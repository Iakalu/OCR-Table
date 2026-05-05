# OCR-Table Deep Learning Pipeline (Local/GitHub Ready)

This repository contains an end-to-end OCR table extraction pipeline for document/table images. It is designed for a Deep Learning coursework/research project and can run in two modes:

- **Lightweight mode**: heuristic/table-line detection + trained `LineSegNet` structure model + PaddleOCR fallback.
- **Full pretrained mode**: Table Transformer detection + Table Transformer structure recognition + PaddleOCR.

The output is a reconstructed table in:

- `CSV`
- `HTML`
- `JSON` with cell coordinates and OCR confidence

---

## What is implemented

### Direction A: Lightweight/custom pipeline

- Table region detection with heuristic and Excel/Sheets screenshot-aware fallback.
- Trainable `LineSegNet` structure model for row/column line segmentation.
- Remote Hugging Face dataset loading for structure training.
- Cell reconstruction from detected row/column geometry.
- PaddleOCR/Tesseract/TrOCR-compatible OCR adapter.
- Streamlit localhost demo.

### Direction B: Full pretrained pipeline

- Table Transformer for table detection.
- Table Transformer for structure recognition.
- PaddleOCR for table-level OCR tokens.
- Token-to-cell assignment by OCR bounding-box center.
- Configurable pipeline modes through YAML.

### Evaluation and debugging

- Cell detection F1 / IoU utilities.
- JSON-based evaluation script.
- OCR backend checker.
- PaddleOCR image debugger.
- OCR debug logs in output directories.

---

## Dataset format support

The training script supports three data sources:

1. **Synthetic tables** generated on the fly.
2. **Remote JSONL manifest** with image URLs and row/column annotations.
3. **Hugging Face dataset**: `katphlab/fintabnet-pubtables-full`.

The selected dataset for this project is:

```text
katphlab/fintabnet-pubtables-full
```

Reason:

- provides table crops;
- provides bounding boxes and category IDs;
- includes row/column/header/spanning-cell labels;
- can be streamed through Hugging Face `datasets` instead of manually downloading files.

Expected category schema:

```text
1 - Table
2 - Column
3 - Row
4 - Column Header
5 - Projected Row Header
6 - Spanning Cell
```

---

## Installation (Windows / VS Code)

Open VS Code terminal in the project root:

```text
C:\Users\ACER\Documents\OCR-Table-main
```

Create and activate virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install OCR stack. Recommended:

```powershell
pip install paddleocr paddlepaddle
```

Optional full pretrained mode:

```powershell
pip install torch torchvision transformers opencv-python
```

Check OCR backend:

```powershell
python check_ocr_backend.py
```

Expected minimum for OCR text extraction:

```text
paddleocr package: True
paddle package: True
```

`tesseract executable: not found in PATH` is acceptable if PaddleOCR is working.

---

## Quick debug mode (recommended first)

Run the pipeline on the sample/synthetic table:

```powershell
python demo.py --config configs/lightweight.yaml --output-dir outputs/demo
```

Run on your own image:

```powershell
python demo.py --config configs/lightweight.yaml --image "path\to\table.png" --output-dir outputs/my_test
```

For the Excel screenshot used during debugging:

```powershell
python demo.py --config configs/lightweight.yaml --image "C:\Users\ACER\OneDrive\Hình ảnh\Screenshots\update-table.png" --output-dir outputs/ocr_excel_debug
```

Check outputs:

```text
outputs/<run_name>/table_0.png
outputs/<run_name>/table_0_ocr_tokens.json
outputs/<run_name>/ocr_debug.log
outputs/<run_name>/result.csv
outputs/<run_name>/result.html
outputs/<run_name>/result.json
```

If `table_0_ocr_tokens.json` does not appear, inspect:

```powershell
notepad outputs\ocr_excel_debug\ocr_debug.log
```

---

## Runbook: Local web demo

Launch Streamlit:

```powershell
streamlit run app.py
```

Open the shown URL, usually:

```text
http://localhost:8501
```

The sidebar allows choosing:

```text
Default
Lightweight
Full pretrained
```

Recommended order:

1. Start with `Lightweight`.
2. Upload an image.
3. Run OCR pipeline.
4. Download `CSV`, `HTML`, or `JSON`.
5. If cells are empty, inspect OCR logs and backend status.

---

## Runbook: Train structure model

### Fast test training

Use this first to verify the whole training pipeline:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 100 --epochs 1 --steps-per-epoch 10 --batch-size 2 --lr 1e-3
```

### Recommended local training

Balanced command for a typical personal laptop/desktop:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 1000 --epochs 4 --steps-per-epoch 80 --batch-size 4 --lr 1e-3
```

### Longer GPU training

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 3000 --epochs 6 --steps-per-epoch 150 --batch-size 8 --lr 8e-4
```

Checkpoint output:

```text
checkpoints/structure_line_cnn.pt
```

After training, run the web app and select:

```text
Lightweight
```

---

## Runbook: Full pretrained mode

Install full dependencies:

```powershell
pip install torch torchvision transformers paddleocr paddlepaddle opencv-python
```

Run CLI:

```powershell
python demo.py --config configs/full_pipeline.yaml --image "path\to\table.png" --output-dir outputs/full_demo
```

Run web:

```powershell
streamlit run app.py
```

Then choose:

```text
Full pretrained
```

Notes:

- First run downloads models from Hugging Face/PaddleOCR.
- Requires internet.
- Uses more memory than lightweight mode.
- If GPU driver is old, PyTorch may fall back to CPU or raise CUDA warnings.

---

## OCR debugging

Check installed backends:

```powershell
python check_ocr_backend.py
```

Debug PaddleOCR on a cropped table image:

```powershell
python debug_paddleocr_image.py --image outputs\ocr_excel_debug\table_0.png --out outputs\ocr_excel_debug\debug_ocr.png
```

Important files:

```text
ocr_debug.log                 # backend loading/runtime messages
table_0_ocr_tokens.json       # OCR text boxes before cell assignment
result.json                   # final cells with text, bbox, row, col
result.csv                    # final table text
```

Common issue:

```text
PaddleOCR load failed: ValueError: Unknown argument: show_log
```

The current OCR adapter handles this by trying multiple PaddleOCR constructors. If this still appears, update the local code or reinstall PaddleOCR.

---

## Evaluation

Evaluate predicted cells against a JSON annotation:

```powershell
python evaluate_pipeline.py --image "path\to\table.png" --target-json "path\to\target_cells.json" --config configs/lightweight.yaml --output-dir outputs/eval
```

Implemented metrics:

- cell precision;
- cell recall;
- cell F1;
- IoU-based matching;
- exact text accuracy.

Recommended report metrics:

- Detection: mAP / precision / recall.
- Structure: cell F1, row/column F1, GriTS or TEDS if extended.
- OCR: exact cell accuracy, CER/WER if ground truth text exists.
- End-to-end: CSV/HTML reconstruction quality.

---

## Local run after downloading checkpoints

Place checkpoints under:

```text
checkpoints/
```

Expected structure checkpoint:

```text
checkpoints/structure_line_cnn.pt
```

Run:

```powershell
python demo.py --config configs/lightweight.yaml --image "path\to\table.png" --output-dir outputs/local_run
```

Or:

```powershell
streamlit run app.py
```

---

## Notes on performance and stability

- `Lightweight` mode is recommended for coursework demos and local debugging.
- `Full pretrained` mode is more realistic but depends on external model downloads.
- PaddleOCR API changes across versions; the adapter tries multiple constructor formats.
- OCR is run at table level first, then tokens are assigned to cells by geometry.
- If `result.csv` has correct grid but empty cells, the problem is OCR, not structure.
- If `result.json` has only one cell, the problem is table structure detection.
- Outputs are written under `outputs/` and ignored by Git.
- Checkpoints are ignored by Git; share large model files through GitHub Release, Google Drive, or Kaggle Output.

---

## Key scripts

```text
app.py                         Streamlit localhost demo
demo.py                        CLI inference demo
train_structure_model.py        Train LineSegNet structure model
evaluate_pipeline.py            Evaluate predicted cells against JSON
check_ocr_backend.py            Check PaddleOCR/Tesseract/TrOCR availability
debug_paddleocr_image.py        Run PaddleOCR directly on one image
```

Core package:

```text
src/table_ocr_pipeline/pipeline/detection.py      Table detection
src/table_ocr_pipeline/pipeline/structure.py      Structure recognition
src/table_ocr_pipeline/pipeline/ocr.py            OCR adapter
src/table_ocr_pipeline/pipeline/reconstruct.py    CSV/HTML/JSON reconstruction
src/table_ocr_pipeline/model/structure_model.py   LineSegNet model
src/table_ocr_pipeline/data/fintabnet_loader.py   Hugging Face dataset loader
```

Configs:

```text
configs/default.yaml
configs/lightweight.yaml
configs/full_pipeline.yaml
```

Docs:

```text
docs/design_vi.md
docs/training_vi.md
docs/running_and_tuning_vi.md
docs/README_vi.md
```

---

## GitHub workflow

For a solo coursework repo, working directly on `main` is acceptable:

```powershell
git status
git add .
git commit -m "Improve table OCR pipeline"
git push
```

Do not commit:

```text
.venv/
outputs/
checkpoints/
data/raw/
*.pt
*.pth
*.onnx
```

These are already covered by `.gitignore`.