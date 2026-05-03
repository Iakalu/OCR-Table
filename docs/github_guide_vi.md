# Huong dan chay va push repo len GitHub

## 1. Kiem tra Python

```bash
python --version
```

Nen dung Python 3.10 tro len.

## 2. Tao virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Neu bi loi execution policy:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

## 3. Cai dependency

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chay demo

```bash
python demo.py --output-dir outputs/demo
```

Mo:

- `outputs/demo/result.csv`
- `outputs/demo/result.html`
- `outputs/demo/result.json`

## 5. Chay web localhost

```bash
streamlit run app.py
```

URL mac dinh:

```text
http://localhost:8501
```

## 6. Chay notebook

```bash
jupyter lab
```

Mo thu muc `notebooks/`.

## 7. Tao repo tren GitHub

1. Vao https://github.com/new
2. Dat ten repo, vi du `table-ocr-dl-pipeline`
3. Chon Public hoac Private
4. Khong can tick README vi repo local da co README
5. Tao repo

## 8. Push code tu local

Trong thu muc project:

```bash
git init
git add .
git commit -m "Initial table OCR deep learning pipeline"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Thay `<username>` va `<repo-name>` bang cua ban.

## 9. Neu GitHub yeu cau login

Dung GitHub CLI:

```bash
gh auth login
```

Hoac dung Personal Access Token khi Git yeu cau password.

## 10. Nhung file khong nen push

Da co `.gitignore` de bo qua:

- `.venv/`
- `outputs/`
- `data/raw/`
- `checkpoints/`
- file weight model `.pt`, `.pth`, `.onnx`, `.ckpt`

