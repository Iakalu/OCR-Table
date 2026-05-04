from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.table_ocr_pipeline.utils.config import load_config
from src.table_ocr_pipeline.pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.pipeline.reconstruct import write_csv, write_html, write_json
from src.table_ocr_pipeline.utils.types import Cell
from src.table_ocr_pipeline.utils.utils import create_synthetic_table


# PATH
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "web"
NOTEBOOK_DIR = ROOT / "notebooks"

CONFIG_OPTIONS = {
    "Default": ROOT / "configs" / "default.yaml",
    "Lightweight": ROOT / "configs" / "lightweight.yaml",
    "Full pretrained": ROOT / "configs" / "full_pipeline.yaml",
}

SYNTHETIC_TEXT = [
    ["Product", "Q1", "Q2", "Growth"],
    ["A", "120", "160", "33%"],
    ["B", "90", "110", "22%"],
    ["C", "210", "205", "-2%"],
]


# PIPELINE
def run_pipeline(image_path: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    pipeline = TableOCRPipeline(config)
    return pipeline.run(image_path, OUTPUT_DIR)


# SYNTHETIC
def fill_synthetic_text():
    cells = []

    for r, row in enumerate(SYNTHETIC_TEXT):
        for c, text in enumerate(row):
            cells.append(
                Cell(
                    row=r,
                    col=c,
                    bbox=(0, 0, 0, 0),
                    text=text,
                    score=1.0,
                )
            )

    write_csv(SYNTHETIC_TEXT, OUTPUT_DIR / "result.csv")
    write_html(SYNTHETIC_TEXT, OUTPUT_DIR / "result.html")
    write_json(cells, OUTPUT_DIR / "result.json")


# HELPERS

def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def output_has_text(result: dict) -> bool:
    import json

    json_path = Path(result["json"])

    if not json_path.exists():
        return False

    cells = json.loads(json_path.read_text(encoding="utf-8"))

    return any(str(c.get("text", "")).strip() for c in cells)


def execute_notebook(notebook_path: Path):
    executed_dir = OUTPUT_DIR / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)

    output_name = notebook_path.stem + "_executed.ipynb"

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output-dir",
        str(executed_dir),
        "--output",
        output_name,
    ]

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    return (
        completed.returncode == 0,
        (completed.stdout + "\n" + completed.stderr).strip(),
        executed_dir / output_name,
    )


# UI
st.set_page_config(page_title="Table OCR Demo", layout="wide")

st.title("Table OCR Pipeline Demo")
st.caption("Detection → Structure → OCR → Reconstruction")


# SIDEBAR
with st.sidebar:
    st.header("Pipeline config")

    config_label = st.selectbox("Mode", list(CONFIG_OPTIONS.keys()))
    selected_config = CONFIG_OPTIONS[config_label]

    st.caption(str(selected_config.relative_to(ROOT)))

    # Notebook runner 
    st.header("Notebook runner")

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    if notebooks:
        selected = st.selectbox(
            "Notebook",
            notebooks,
            format_func=lambda p: p.name,
        )

        if st.button("Execute notebook"):
            with st.spinner(f"Executing {selected.name}"):
                ok, log, out_path = execute_notebook(selected)

            if ok:
                st.success("Notebook executed")
            else:
                st.error("Notebook failed")

            if out_path.exists():
                st.download_button(
                    "Download executed notebook",
                    read_bytes(out_path),
                    file_name=out_path.name,
                )

            if log:
                st.code(log[-4000:])
    else:
        st.info("No notebooks found")


# MAIN LAYOUT
left, right = st.columns([0.45, 0.55])

# LEFT (INPUT)
with left:
    uploaded = st.file_uploader("Upload table image", type=["png", "jpg", "jpeg"])
    use_sample = st.button("Use sample table")

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())

            st.session_state["image_path"] = str(Path(f.name))
            st.session_state["is_sample"] = False

    elif use_sample:
        sample_path = create_synthetic_table(OUTPUT_DIR / "synthetic.png")

        st.session_state["image_path"] = str(sample_path)
        st.session_state["is_sample"] = True

    image_path = Path(st.session_state["image_path"]) if "image_path" in st.session_state else None
    is_sample = bool(st.session_state.get("is_sample", False))

    if image_path:
        st.image(Image.open(image_path), caption="Input image", use_container_width=True)

        if st.button("Run OCR pipeline", type="primary"):
            with st.spinner("Running pipeline..."):
                result = run_pipeline(image_path, selected_config)

                if is_sample:
                    fill_synthetic_text()

                st.session_state["result"] = result


# RIGHT (OUTPUT)
with right:
    result = st.session_state.get("result")

    if not result:
        st.info("Upload an image or use sample → Run pipeline")
    else:
        st.success(f"Done: {result['tables']} table(s), {result['cells']} cell(s)")

        if not output_has_text(result):
            st.warning("OCR text is empty. Install PaddleOCR or enable full pretrained config.")

        crop_path = OUTPUT_DIR / "table_0.png"

        if crop_path.exists():
            st.image(Image.open(crop_path), caption="Detected table")

        csv_path = Path(result["csv"])
        html_path = Path(result["html"])
        json_path = Path(result["json"])

        st.download_button("Download CSV", read_bytes(csv_path), file_name="result.csv")
        st.download_button("Download HTML", read_bytes(html_path), file_name="result.html")
        st.download_button("Download JSON", read_bytes(json_path), file_name="result.json")

        if html_path.exists():
            st.subheader("Preview")

            components.html(
                html_path.read_text(encoding="utf-8"),
                height=320,
                scrolling=True,
            )