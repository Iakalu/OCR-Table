# Thiet ke mo hinh day du cho OCR bang bieu

Tai thoi diem nay repo co 2 mode:

- `lightweight`: chay nhanh, co fallback heuristic va LineSegNet.
- `full_pipeline`: dung Table Transformer cho detection/structure va PaddleOCR cho text.

## 1. Full Pipeline

```text
Image/PDF page
  -> Preprocess
  -> Table Detection: Table Transformer detection
  -> Table Crop
  -> Structure Recognition: Table Transformer structure
  -> Cell Grid Reconstruction
  -> Cell OCR: PaddleOCR
  -> Post-processing
  -> CSV / HTML / JSON
```

## 2. Detection model

Config:

```yaml
detection:
  backend: table_transformer
  model_name: microsoft/table-transformer-detection
  confidence_threshold: 0.65
```

Vai tro:

- tim bbox cua tung bang trong trang;
- giam nhieu cho structure model vi chi xu ly crop cua bang.

Neu fine-tune:

- Dataset: PubTables-1M / TableBank.
- Metric: mAP@IoU, precision, recall.
- Hyperparameters: learning rate `1e-5`, image size `1024-1333`, batch size tuy GPU.

## 3. Structure model

Config:

```yaml
structure:
  backend: table_transformer
  model_name: microsoft/table-transformer-structure-recognition
  structure_threshold: 0.65
```

Model predict:

- table row;
- table column;
- table column header;
- projected row header;
- spanning cell.

Repo reconstruct cell bằng giao điểm row-box và column-box:

```text
cell_bbox = intersection(row_bbox, column_bbox)
```

Huong nang cap tiep:

- xu ly `spanning_cell` de tao `row_span` va `col_span`;
- gan header/body role vao cell;
- dung GriTS/TEDS de evaluate structure.

## 4. OCR model

Config:

```yaml
ocr:
  backend: paddleocr
  lang: en
  cell_padding: 4
```

Vai tro:

- crop tung cell;
- doc text;
- tra ve text va confidence.

Neu bang tieng Viet:

- doi OCR backend/lang sang model ho tro Vietnamese;
- them post-processing cho dau tieng Viet, number, date, currency.

## 5. Reconstruction

Output chinh:

- JSON: dung de debug va evaluate;
- HTML: giu cau truc bang tot hon CSV;
- CSV: dung khi bang khong merged cell.

## 6. Cau hinh chay

Chay nhe:

```powershell
python demo.py --config configs/lightweight.yaml --output-dir outputs/demo_light
```

Chay full model:

```powershell
python demo.py --config configs/full_pipeline.yaml --image path\to\table.png --output-dir outputs/demo_full
```

Web localhost hien dang load `configs/default.yaml`. Neu muon web dung full pipeline, sua trong `app.py`:

```python
config = load_config(ROOT / "configs" / "full_pipeline.yaml")
```

Hoac doi `configs/default.yaml` sang noi dung cua `configs/full_pipeline.yaml`.

## 7. Luu y thuc te

Full pipeline can cac package nang:

```powershell
pip install torch torchvision transformers paddleocr paddlepaddle opencv-python
```

Lan dau chay se tai pretrained model tu Hugging Face/PaddleOCR nen can internet.

## 8. Metrics nen bao cao

- Detection: mAP, precision, recall.
- Structure: cell F1, row/column F1, GriTS/TEDS.
- OCR: CER/WER, exact cell text accuracy.
- End-to-end: exact CSV/HTML match voi ground truth.
