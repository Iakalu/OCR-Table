# Table OCR Deep Learning Pipeline

Du an mau cho bai tap Deep Learning: OCR bang bieu tu anh/PDF scan.

Pipeline:

```text
Input image/PDF
  -> Preprocess
  -> Table Detection
  -> Table Structure Recognition
  -> Cell OCR
  -> Post-processing / Reconstruction
  -> CSV / HTML / JSON
```

Repo nay co 2 muc tieu:

1. Giai thich kien truc va cach xay dung model OCR bang bieu theo tung buoc.
2. Co code demo/web localhost/notebook de chay va trinh bay tren GitHub.

## 1. Vi sao chon OCR bang bieu?

OCR thong thuong chi tra ve text. Voi bang bieu, ta can biet text thuoc hang nao, cot nao, o nao, co merged cell hay khong. Do do bai toan dung hon la table understanding.

Ung dung:

- trich xuat bang tu bao cao tai chinh;
- doc hoa don, phieu thu, phieu xuat kho;
- trich xuat ket qua xet nghiem/y te;
- chuyen bang trong PDF scan thanh CSV/Excel;
- document AI cho van ban hanh chinh.

## 2. Kien truc de xuat

### Buoc 1: Preprocess

- convert PDF page thanh image 200-300 DPI;
- deskew anh nghieng;
- denoise, normalize contrast;
- resize giu aspect ratio.

### Buoc 2: Table Detection

Input: anh trang tai lieu.

Output: bounding box cua tung bang.

Model co the dung:

- YOLO: nhanh, de train/deploy;
- Faster R-CNN: baseline detection tot;
- DETR/Table Transformer: phu hop de trinh bay research;
- CascadeTabNet/Mask R-CNN: tot khi can mask/structure.

### Buoc 3: Table Structure Recognition

Input: crop cua bang.

Output: row, column, cell, spanning cell, header/body.

Model co the dung:

- Table Transformer structure recognition;
- TSRFormer;
- Graph/neural parser tren OCR word boxes;
- image-to-HTML model tren PubTabNet.

### Buoc 4: Cell Text OCR

Input: crop tung cell.

Output: text + confidence.

Model co the dung:

- PaddleOCR: de dung, manh cho demo;
- CRNN: nhe, de train lai;
- TrOCR: Transformer OCR, tot nhung nang;
- Tesseract: baseline nhanh cho anh sach.

### Buoc 5: Reconstruction

- sort rows/columns theo toa do;
- match OCR text vao cell;
- xu ly row span / column span;
- xuat HTML de giu structure;
- xuat CSV khi bang don gian;
- xuat JSON de debug/model evaluation.

## 3. Cau truc project

```text
.
|-- app.py                         # web localhost bang Streamlit
|-- demo.py                        # demo CLI
|-- requirements.txt
|-- pyproject.toml
|-- configs/
|   `-- default.yaml
|-- data/
|   `-- README.md                  # cach ket noi dataset
|-- docs/
|   |-- github_guide_vi.md
|   |-- model_design_vi.md
|   `-- tuning_guide_vi.md
|-- notebooks/
|   |-- 01_architecture_demo.ipynb
|   |-- 02_training_skeleton.ipynb
|   `-- 03_tuning_and_error_analysis.ipynb
`-- src/
    `-- table_ocr_pipeline/
        |-- config.py
        |-- detection.py
        |-- ocr.py
        |-- pipeline.py
        |-- reconstruct.py
        |-- structure.py
        |-- types.py
        `-- utils.py
```

## 4. Cai dat

Tao moi truong ao:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Neu bi chan script policy tren Windows:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

Cai dependency:

```bash
pip install -r requirements.txt
```

## 5. Chay demo CLI

```bash
python demo.py --output-dir outputs/demo
```

Ket qua:

- `outputs/demo/synthetic_table.png`
- `outputs/demo/table_0.png`
- `outputs/demo/result.csv`
- `outputs/demo/result.html`
- `outputs/demo/result.json`

## 6. Chay web localhost

```bash
streamlit run app.py
```

Mo browser tai URL Streamlit hien ra, thuong la:

```text
http://localhost:8501
```

Web app co the:

- upload anh bang;
- dung anh bang synthetic mau;
- chay pipeline;
- xem crop bang;
- tai CSV/HTML/JSON;
- chay notebook trong sidebar bang `jupyter nbconvert`.

Neu muon chay full model pretrained, cai them:

```bash
pip install torch torchvision transformers paddleocr paddlepaddle opencv-python
```

Sau do doi config trong `app.py` tu `configs/default.yaml` sang `configs/full_pipeline.yaml`, hoac chay CLI:

```bash
python demo.py --config configs/full_pipeline.yaml --image path/to/table.png --output-dir outputs/full_demo
```

Neu output co bang nhung cac o rong, kiem tra OCR:

```bash
python check_ocr_backend.py
```

Huong dan chi tiet xem `docs/ocr_setup_vi.md`.

## 7. Chay notebook

```bash
jupyter lab
```

Mo thu muc `notebooks/`:

- `01_architecture_demo.ipynb`: kien truc va demo;
- `02_training_skeleton.ipynb`: skeleton train detection/structure;
- `03_tuning_and_error_analysis.ipynb`: tinh chinh tham so va phan tich loi.

## 8. Nang cap sang model Deep Learning that

Ban demo mac dinh co fallback heuristic de chay nhanh. De dung model that:

```bash
pip install torch torchvision transformers paddleocr paddlepaddle opencv-python
```

Sau do sua `configs/default.yaml`:

```yaml
detection:
  backend: table_transformer
ocr:
  backend: paddleocr
```

Model detection/structure nen fine-tune tren PubTables-1M. OCR co the dung PaddleOCR truoc, sau do fine-tune CRNN/TrOCR neu domain dac thu.

## 8.1 Train structure model local

Repo co san mot CNN nho `LineSegNet` de train table structure tren synthetic table. Model nay hoc predict 2 mask: duong ke doc va duong ke ngang.

Train nhanh tren CPU:

```bash
python train_structure_model.py --epochs 8 --steps-per-epoch 120 --batch-size 8
```

Train nhanh hon neu may yeu:

```bash
python train_structure_model.py --epochs 3 --steps-per-epoch 40 --batch-size 4
```

Checkpoint se luu tai:

```text
checkpoints/structure_line_cnn.pt
```

Sau khi train, `configs/default.yaml` da de:

```yaml
structure:
  backend: auto
  checkpoint_path: checkpoints/structure_line_cnn.pt
```

Nen pipeline/web localhost se tu dung model nay neu checkpoint ton tai; neu khong co checkpoint thi fallback ve heuristic.

## 8.2 Train structure model truc tiep tu link

Neu ban khong muon tai dataset thu cong vao `data/raw/`, tao mot remote manifest `.jsonl`. Moi dong gom link anh va annotation line:

```json
{"image_url":"https://domain.com/table_001.png","vertical_lines":[0.05,0.25,0.50,0.75,0.95],"horizontal_lines":[0.08,0.30,0.52,0.74,0.92]}
```

Train truc tiep tu manifest URL:

```bash
python train_structure_model.py --data-source manifest-url --manifest-url "https://your-domain.com/table_structure_manifest.jsonl" --epochs 5 --steps-per-epoch 100 --batch-size 4
```

Chi tiet xem `docs/remote_dataset_training_vi.md`.

## 8.3 Dataset chinh duoc chon cho bai

Dataset minh chon cho project nay la Hugging Face `katphlab/fintabnet-pubtables-full`.

Ly do:

- co anh table crop;
- co bounding boxes cho `table`, `column`, `row`, `column_header`, `projected_row_header`, `spanning_cell`;
- dung schema gan voi Table Transformer/TATR;
- co the doc bang Hugging Face `datasets` thay vi tai thu cong vao `data/raw/`.

Train structure model truc tiep tu Hugging Face:

```bash
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 2000 --epochs 5 --steps-per-epoch 100 --batch-size 4
```

Neu may yeu, chay ban nho:

```bash
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 200 --epochs 2 --steps-per-epoch 20 --batch-size 2
```

Dataset nay co config rieng `train`, `val`, `test`. Script se tu dung `--hf-split` lam config mac dinh. Neu muon chi ro:

```bash
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-config train --hf-split train --max-remote-samples 200 --epochs 2 --steps-per-epoch 20 --batch-size 2
```

## 9. Dataset goi y

- PubTables-1M: detection + structure recognition.
- TableBank: table detection.
- PubTabNet: image-to-HTML table recognition.
- FinTabNet: bang tai chinh.
- ICDAR 2019 cTDaR: benchmark kho cho scan/historical documents.

Chi tiet xem `data/README.md`.

## 10. Push len GitHub

Xem huong dan chi tiet trong `docs/github_guide_vi.md`.

Lenh nhanh:

```bash
git init
git add .
git commit -m "Initial table OCR deep learning pipeline"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```
