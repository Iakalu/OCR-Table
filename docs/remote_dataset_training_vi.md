# Train structure model truc tiep tu link

Muc tieu: khong can tai dataset thu cong vao `data/raw/`. Script train se doc annotation tu mot file JSONL remote va tai tung anh vao RAM khi train.

## 1. Vi sao can annotation?

Structure model can hoc vi tri hang/cot. Vi vay moi sample can co:

- `image_url`: link anh bang;
- `vertical_lines`: vi tri cac duong/cot;
- `horizontal_lines`: vi tri cac duong/hang.

Neu chi co anh ma khong co annotation, model khong co nhan de hoc.

## 2. Dinh dang remote manifest

Manifest la file `.jsonl`, moi dong la mot JSON object:

```json
{"image_url":"https://domain.com/table_001.png","vertical_lines":[0.05,0.25,0.50,0.75,0.95],"horizontal_lines":[0.08,0.30,0.52,0.74,0.92],"line_mask_width":3}
```

`vertical_lines` va `horizontal_lines` co the dung:

- toa do normalized tu `0.0` den `1.0`;
- hoac toa do pixel.

Vi du local nam o `data/remote_manifest_example.jsonl`.

## 3. Train tu manifest URL

```bash
python train_structure_model.py ^
  --data-source manifest-url ^
  --manifest-url https://your-domain.com/table_structure_manifest.jsonl ^
  --epochs 5 ^
  --steps-per-epoch 100 ^
  --batch-size 4
```

PowerShell mot dong:

```powershell
python train_structure_model.py --data-source manifest-url --manifest-url "https://your-domain.com/table_structure_manifest.jsonl" --epochs 5 --steps-per-epoch 100 --batch-size 4
```

Checkpoint:

```text
checkpoints/structure_line_cnn.pt
```

Sau do chay:

```powershell
python demo.py --output-dir outputs/demo_remote_trained
streamlit run app.py
```

## 4. Hugging Face streaming

Neu dataset co san tren Hugging Face va duoc ho tro `datasets.load_dataset(..., streaming=True)`, ban co the streaming khong can tai het ve may.

Theo tai lieu Hugging Face Datasets, `streaming=True` cho phep du lieu duoc tai dan khi lap qua dataset, huu ich khi dataset rat lon hoac khong du dung luong dia.

Vi du:

```python
from datasets import load_dataset

ds = load_dataset("liminghao1630/TableBank", split="train", streaming=True)
for sample in ds:
    ...
```

Luu y: moi dataset co schema khac nhau. Muon train structure line model, sample van can co annotation de tao `vertical_lines`/`horizontal_lines` hoac cell boxes.

## 5. Dataset duoc chon cho project

Project nay chon:

```text
katphlab/fintabnet-pubtables-full
```

Ly do:

- Co san tren Hugging Face.
- Co anh table crop.
- Co `boxes` va `category_ids`.
- Category schema:

```text
1 - Table
2 - Column
3 - Row
4 - Column Header
5 - Projected Row Header
6 - Spanning Cell
```

Script `train_structure_model.py` dung class `Column` de tao vertical line mask va class `Row` de tao horizontal line mask.

Train nhanh:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 200 --epochs 2 --steps-per-epoch 20 --batch-size 2
```

Train tot hon:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 2000 --epochs 5 --steps-per-epoch 100 --batch-size 4
```

## 6. Dataset nao khac phu hop?

- TableBank tren Hugging Face co the dung cho table detection, nhung structure annotation co the khong du de train line/cell model truc tiep.
- PubTables-1M co annotation structure tot, nhung ban Hugging Face hien tai chu yeu cung cap file tar.gz; neu khong muon tai ve, can tao manifest URL rieng tu storage/cloud.
- PubTabNet phu hop image-to-HTML, nhung can parser HTML de suy grid/cell lines.

## 7. Khuyen nghi cho bai cua ban

De bao cao dep va chay duoc:

1. Train synthetic truoc de chung minh model train duoc.
2. Tao 20-100 sample remote manifest tren GitHub raw/Google Cloud/S3.
3. Train bang `--data-source manifest-url`.
4. Neu co thoi gian, chuyen annotation PubTables-1M thanh manifest JSONL va host manifest + image URLs.
