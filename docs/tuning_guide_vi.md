# Huong dan tinh chinh tham so

File chinh: `configs/default.yaml`.

## Detection

```yaml
detection:
  confidence_threshold: 0.55
  nms_iou: 0.50
  image_size: 1024
```

Tinh huong:

- Miss bang: giam `confidence_threshold`, tang `image_size`.
- Nhan nham text block la bang: tang `confidence_threshold`.
- Hai bang gan nhau bi gop: giam `nms_iou`.
- Mot bang bi tach thanh nhieu bbox: tang `nms_iou`.

## Structure

```yaml
structure:
  structure_threshold: 0.50
  merge_tolerance: 8
  min_cell_width: 24
  min_cell_height: 16
```

Tinh huong:

- Tach qua nhieu cot/hang: tang `min_cell_width`, `min_cell_height`.
- Gop nham cot/hang: giam `merge_tolerance`.
- Anh scan nghieng/mo: tang `merge_tolerance`, bat deskew/preprocess.
- Bang day dac: giam `min_cell_width`, `min_cell_height`.

## OCR

```yaml
ocr:
  lang: en
  confidence_threshold: 0.50
  cell_padding: 3
```

Tinh huong:

- Mat chu sat border: tang `cell_padding`.
- OCR nham border thanh ky tu: giam `cell_padding` hoac remove border truoc OCR.
- Tieng Viet: doi `lang` sang backend ho tro `vi`.
- Bang so lieu: them post-processing normalize number/date/currency.

## Theo loai bang

Bang co duong ke ro:

- heuristic line detection co the dung lam fallback;
- `merge_tolerance` 4-8;
- `cell_padding` 2-4.

Bang khong co duong ke:

- can model structure DL;
- tang `image_size`;
- ket hop OCR word boxes de suy row/column.

Bang scan xau:

- augmentation khi train: blur, rotation, perspective, brightness;
- preprocess: deskew, denoise, contrast;
- tang `merge_tolerance` va `cell_padding`.

Bang tai chinh:

- fine-tune tren FinTabNet;
- post-process currency, percentage, negative number;
- dung HTML/JSON thay vi ep CSV qua som neu co merged headers.

