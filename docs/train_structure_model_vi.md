# Train structure model

Muc tieu: train mot model CNN nho de nhan dang cau truc bang co duong ke.

Model: `LineSegNet`

Input:

- anh grayscale kich thuoc 256x256;
- synthetic table co hang/cot random, line width random, noise va fake text block.

Output:

- channel 0: mask duong ke doc;
- channel 1: mask duong ke ngang.

Loss:

- `BCEWithLogitsLoss`.

## Cai dat

```bash
pip install -r requirements.txt
```

Neu loi cai torch tren Windows, cai theo lenh tren trang PyTorch: https://pytorch.org/get-started/locally/

## Train nhanh

```bash
python train_structure_model.py --epochs 3 --steps-per-epoch 40 --batch-size 4
```

## Train tot hon

```bash
python train_structure_model.py --epochs 8 --steps-per-epoch 120 --batch-size 8
```

## Output

```text
checkpoints/structure_line_cnn.pt
```

## Chay demo sau khi train

```bash
python demo.py --output-dir outputs/demo_trained
```

Hoac web:

```bash
streamlit run app.py
```

## Tinh chinh

Trong `configs/default.yaml`:

```yaml
structure:
  backend: auto
  checkpoint_path: checkpoints/structure_line_cnn.pt
  line_threshold: 0.45
  merge_tolerance: 8
```

- Neu model tao qua nhieu line: tang `line_threshold`.
- Neu model bo sot line: giam `line_threshold`.
- Neu nhieu line gan nhau bi tach rieng: tang `merge_tolerance`.
- Neu cot/hang bi gop nham: giam `merge_tolerance`.

## Gioi han

Model synthetic nay phu hop de demo train structure. No chua thay the hoan toan Table Transformer/PubTables-1M cho bang that phuc tap, khong duong ke, merged cells, anh scan xau.
