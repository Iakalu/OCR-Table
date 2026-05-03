# Thiet ke model OCR bang bieu

## Bai toan

Input la anh/PDF page co mot hoac nhieu bang. Output la du lieu co cau truc: bbox bang, hang, cot, cell, text tung cell, va file CSV/HTML/JSON.

## Pipeline Deep Learning

### 1. Preprocess

Muc tieu la giam loi do anh scan:

- resize theo `max_side`;
- deskew neu anh bi nghieng;
- denoise neu anh mo/nhieu;
- normalize contrast.

### 2. Table Detection

Model de xuat:

- YOLO neu can toc do;
- Faster R-CNN neu can baseline on dinh;
- DETR/Table Transformer neu muon huong research hien dai.

Loss voi DETR/Table Transformer:

- classification loss;
- L1 bbox loss;
- GIoU loss;
- Hungarian matching de gan query voi object.

### 3. Table Structure Recognition

Hai huong chinh:

- detect row/column/cell/spanning cell;
- decode truc tiep HTML token.

Huong nen chon cho project nay la detect structure objects bang Table Transformer, sau do reconstruct grid bang geometry.

### 4. Cell OCR

Chon PaddleOCR lam baseline vi:

- co san text detection + recognition;
- chay nhanh cho demo;
- ho tro nhieu ngon ngu;
- de thay bang CRNN/TrOCR sau nay.

### 5. Reconstruction

Sau khi co cell bbox va OCR text:

1. sort rows theo truc y;
2. sort columns theo truc x;
3. gan text vao cell bang center point/IoU;
4. xu ly spanning cell;
5. xuat HTML de giu cau truc;
6. xuat CSV cho bang khong co merged cell.

## Tai sao pipeline modular tot?

- De debug tung buoc.
- De thay model rieng le khi domain thay doi.
- De giai thich trong bao cao ro rang.
- De demo ngay ca khi chua train end-to-end.

## Paper/dataset nen trinh bay

- DeepDeSRT: CNN cho table detection va structure recognition.
- TableBank: dataset lon sinh tu Word/LaTeX.
- CascadeTabNet: Cascade Mask R-CNN cho table detection/structure.
- PubTables-1M/Table Transformer: DETR cho table detection va structure recognition.
- GriTS: metric danh gia table structure theo grid.
- PubTabNet: image-to-HTML table recognition.

