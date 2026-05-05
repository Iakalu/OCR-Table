# Training (structure line CNN) va remote dataset

## 1. Train structure line CNN (LineSegNet)

Muc tieu: train mot CNN nho de nhan dang duong ke doc/ngang (phu hop bang co duong ke).

Train nhanh:

```powershell
python train_structure_model.py --epochs 3 --steps-per-epoch 40 --batch-size 4
```

Train tot hon:

```powershell
python train_structure_model.py --epochs 8 --steps-per-epoch 120 --batch-size 8
```

Checkpoint:

```text
checkpoints/structure_line_cnn.pt
```

Sau do chay lai demo/web (config `default.yaml` se tu dung checkpoint neu ton tai).

## 2. Train tu remote manifest (JSONL)

Manifest `.jsonl`, moi dong la:

```json
{"image_url":"https://domain.com/table_001.png","vertical_lines":[0.05,0.25,0.50,0.75,0.95],"horizontal_lines":[0.08,0.30,0.52,0.74,0.92],"line_mask_width":3}
```

Train:

```powershell
python train_structure_model.py --data-source manifest-url --manifest-url "https://your-domain.com/table_structure_manifest.jsonl" --epochs 5 --steps-per-epoch 100 --batch-size 4
```

## 3. Streaming dataset tu Hugging Face (tuy chon)

Neu dataset ho tro streaming:

```python
from datasets import load_dataset
ds = load_dataset("liminghao1630/TableBank", split="train", streaming=True)
for sample in ds:
    ...
```

Repo co san option streaming cho `katphlab/fintabnet-pubtables-full` trong `train_structure_model.py`.

