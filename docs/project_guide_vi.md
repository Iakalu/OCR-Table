# Huong dan project OCR-Table

Tai lieu nay gom cac noi dung thiet ke, cai dat, chay demo, tuning va training cho project OCR bang bieu.

---

## 1. Thiet ke pipeline

### Bai toan

Input: anh/PDF page co mot hoac nhieu bang.  
Output: du lieu co cau truc, gom bbox bang/hang/cot/cell, text tung cell, va file `CSV/HTML/JSON`.

### Pipeline tong quan

```text
Image/PDF
  -> Preprocess
  -> Table Detection
  -> Table Crop
  -> Table Structure Recognition
  -> OCR
  -> Reconstruction
  -> CSV / HTML / JSON
```

### Mode Lightweight

Dung cho demo local va anh screenshot bang co duong ke:

- Detection: heuristic + Excel/Sheets color-aware fallback.
- Structure: LineSegNet neu co checkpoint, fallback line projection.
- OCR: PaddleOCR uu tien, fallback Tesseract/TrOCR neu cai dat.

Config:

```text
configs/lightweight.yaml
preset `presets.excel_screenshot` trong `configs/full_pipeline.yaml`
```

### Mode Full pretrained

Dung cho pipeline gan production/research hon:

- Detection: `microsoft/table-transformer-detection`
- Structure: `microsoft/table-transformer-structure-recognition`
- OCR: PaddleOCR

Config:

```text
configs/full_pipeline.yaml
```

### Reconstruction

- JSON: debug/evaluate, co row/col/bbox/text/score.
- HTML: xem bang de hon CSV.
- CSV: phu hop bang khong/it merged cell.

---

## 2. Cai dat va chay

### Tao moi truong

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu PowerShell chan activate:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

### Cai OCR backend

Khuyen dung PaddleOCR:

```powershell
python -m pip install paddleocr paddlepaddle
python check_ocr_backend.py
```

Can thay:

```text
paddleocr package : O
paddle package    : O
usable            : O
```

Tesseract la optional. Neu `tesseract executable: not found in PATH` nhung PaddleOCR usable thi van du.

### Chay CLI

```powershell
python demo.py --config configs/lightweight.yaml --output-dir outputs/demo
```

Anh Excel/screenshot:

```powershell
python demo.py --config preset `presets.excel_screenshot` trong `configs/full_pipeline.yaml` --image "path\to\table.png" --output-dir outputs/excel_test
```

### Chay web localhost

```powershell
streamlit run app.py
```

Trong sidebar chon mode:

```text
Default
Lightweight
Lightweight
Full pretrained
```

---

## 3. Debug OCR va output rong

### Neu `result.json` la `[]`

Pipeline khong tao duoc cell. Nguyen nhan thuong gap:

- Detection khong tim thay bang.
- Structure model khong tach duoc row/column.
- Chon sai config cho dang anh.

Voi screenshot Excel, dung:

```text
preset `presets.excel_screenshot` trong `configs/full_pipeline.yaml`
```

### Neu CSV co grid nhung cell rong

Structure da chay, OCR chua doc duoc text.

Kiem tra:

```powershell
python check_ocr_backend.py
```

Debug PaddleOCR tren crop bang:

```powershell
python debug_paddleocr_image.py --image outputs\excel_test\table_0.png --out outputs\excel_test\debug_ocr.png
```

File can xem:

```text
outputs/<run>/ocr_debug.log
outputs/<run>/table_0_ocr_tokens.json
outputs/<run>/result.json
outputs/<run>/result.csv
```

---

## 4. Tuning nhanh

### Miss table

- Giam `detection.confidence_threshold`.
- Tang `preprocess.max_side`.
- Voi screenshot Excel, dung `detection.backend: heuristic`.

### Sai hang/cot

- Tang/giam `structure.merge_tolerance`.
- Dieu chinh `structure.min_cell_width` va `structure.min_cell_height`.
- Voi bang co duong ke ro, dung `structure.backend: heuristic` hoac `line_cnn`.

### OCR sai / mat chu

- Tang `ocr.cell_padding` neu chu sat border.
- Tat `binarize` neu chu mau/nen mau bi mat thong tin.
- Dung table-level OCR tokens thay vi OCR tung cell khi chu qua nho.

---

## 5. Training structure model

### Synthetic training

Nhanh:

```powershell
python train_structure_model.py --epochs 3 --steps-per-epoch 40 --batch-size 4
```

Tot hon:

```powershell
python train_structure_model.py --epochs 8 --steps-per-epoch 120 --batch-size 8
```

Checkpoint:

```text
checkpoints/structure_line_cnn.pt
```

### Hugging Face dataset

Dataset chinh:

```text
katphlab/fintabnet-pubtables-full
```

Test nhanh:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 100 --epochs 1 --steps-per-epoch 10 --batch-size 2 --lr 1e-3
```

Training khuyen dung:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 1000 --epochs 4 --steps-per-epoch 80 --batch-size 4 --lr 1e-3
```

### Remote manifest JSONL

Moi dong:

```json
{"image_url":"https://domain.com/table_001.png","vertical_lines":[0.05,0.25,0.50,0.75,0.95],"horizontal_lines":[0.08,0.30,0.52,0.74,0.92],"line_mask_width":3}
```

Train:

```powershell
python train_structure_model.py --data-source manifest-url --manifest-url "https://your-domain.com/table_structure_manifest.jsonl" --epochs 5 --steps-per-epoch 100 --batch-size 4
```

---

## 6. Evaluation

```powershell
python evaluate_pipeline.py --image "path\to\table.png" --target-json "path\to\target_cells.json" --config configs/lightweight.yaml --output-dir outputs/eval
```

Metrics hien co:

- cell precision;
- cell recall;
- cell F1;
- IoU matching;
- exact text accuracy.

---

## 7. File quan trong

```text
app.py                                      Streamlit app
demo.py                                     CLI inference
train_structure_model.py                    Train LineSegNet
evaluate_pipeline.py                        Evaluate output
check_ocr_backend.py                        Check OCR backend
debug_paddleocr_image.py                    Debug PaddleOCR
src/table_ocr_pipeline/pipeline/*.py        Pipeline core
src/table_ocr_pipeline/model/structure_model.py
configs/*.yaml                              Runtime configs
```