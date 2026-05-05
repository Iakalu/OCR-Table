# Thiet ke pipeline OCR bang bieu

## Bai toan

Input: anh/PDF page co mot hoac nhieu bang.  
Output: du lieu co cau truc (bbox bang/hang/cot/cell + text tung cell) va file `CSV/HTML/JSON`.

## Pipeline tong quan

```text
Image/PDF -> Preprocess -> Table Detection -> Table Crop -> Structure Recognition -> OCR -> Reconstruction -> CSV/HTML/JSON
```

## Full pipeline (pretrained)

Repo co the chay mode full pretrained (tham khao `configs/full_pipeline.yaml`):

- **Detection**: `microsoft/table-transformer-detection`
- **Structure**: `microsoft/table-transformer-structure-recognition`
- **OCR**: PaddleOCR (uu tien), fallback Tesseract/TrOCR (tuy cai dat)

### Detection

Vai tro:
- tim bbox bang trong anh trang;
- giam nhieu cho structure model vi chi xu ly crop cua bang.

Neu miss bang:
- giam `detection.confidence_threshold`
- tang `preprocess.max_side`

### Structure recognition

Model predict cac thanh phan nhu row/column/... Sau do repo reconstruct cell bang giao diem:

```text
cell_bbox = intersection(row_bbox, column_bbox)
```

Huong nang cap tiep (neu can):
- xu ly `spanning_cell` de tao `row_span/col_span`;
- danh gia structure bang GriTS/TEDS.

### Cell OCR

Moi cell duoc crop va OCR:
- token-based OCR (neu co token bbox)
- fallback OCR tung cell

### Reconstruction

Output:
- JSON: debug/evaluate
- HTML: giu structure tot hon CSV
- CSV: phu hop khi bang it/khong merged cell

