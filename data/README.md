# Data / Dataset setup

Khong commit dataset vao GitHub. Cac thu muc du lieu lon nen dat trong `data/raw/` (da duoc ignore trong `.gitignore`).

## 1. Local datasets (neu ban muon tai ve)

Goi y cau truc:

```text
data/raw/
  pubtables1m/        # neu fine-tune Table Transformer
  tablebank/
  pubtabnet/
  fintabnet/
```

## 2. Train structure model tu Hugging Face (khong can tai dataset ve)

Repo ho tro streaming dataset `katphlab/fintabnet-pubtables-full` trong `train_structure_model.py`:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 2000 --epochs 5 --steps-per-epoch 100 --batch-size 4
```

## 3. Remote manifest JSONL (khong can luu dataset local)

Neu ban co it sample va muon host tren internet (GitHub raw / S3 / GCS), tao file `.jsonl`:

- `image_url`: URL anh bang (png/jpg)
- `vertical_lines`, `horizontal_lines`: toa do line (normalized 0..1 hoac pixel)
- `line_mask_width`: (tuy chon) do day line mask

Xem vi du: `data/remote_manifest_example.jsonl`

Train:

```powershell
python train_structure_model.py --data-source manifest-url --manifest-url "https://your-domain.com/table_structure_manifest.jsonl" --epochs 5 --steps-per-epoch 100 --batch-size 4
```

## 4. Tai lieu lien quan

Xem them: `docs/training_vi.md`
