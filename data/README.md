# Dataset setup

Khong commit dataset vao GitHub. Hay dat dataset trong `data/raw/`.

## PubTables-1M

- Link: https://github.com/microsoft/table-transformer
- Dung cho: table detection, structure recognition, functional analysis.
- Phu hop nhat neu ban muon fine-tune Table Transformer.

Goi y:

```text
data/raw/pubtables1m/
  images/
  train/
  val/
  test/
  words/
```

## TableBank

- Link: https://github.com/doc-analysis/TableBank
- Dung cho: table detection voi document sinh tu Word/LaTeX.

## PubTabNet

- Link: https://github.com/ibm-aur-nlp/PubTabNet
- Dung cho: image-to-HTML table structure recognition.

## FinTabNet

- Dung cho: bang tai chinh, header phuc tap, merged cells.

## ICDAR 2019 cTDaR

- Link: https://zenodo.org/records/2649217
- Dung cho: benchmark kho voi scan/historical documents.

## Chien luoc thuc hien bai tap

Dataset chinh da chon cho repo:

```text
katphlab/fintabnet-pubtables-full
```

Train truc tiep tu Hugging Face:

```powershell
python train_structure_model.py --data-source hf-fintabnet-pubtables --hf-split train --max-remote-samples 2000 --epochs 5 --steps-per-epoch 100 --batch-size 4
```

Chien luoc:

1. Demo ban dau bang synthetic table.
2. Train structure model bang `katphlab/fintabnet-pubtables-full`.
3. Dung `boxes` + `category_ids` de tao mask row/column.
4. Sau khi train, checkpoint nam tai `checkpoints/structure_line_cnn.pt`.
5. Chay lai web/demo de dung model structure da train.
