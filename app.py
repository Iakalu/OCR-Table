from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.table_ocr_pipeline.config import load_config
from src.table_ocr_pipeline.pipeline import TableOCRPipeline
from src.table_ocr_pipeline.reconstruct import write_csv, write_html, write_json
from src.table_ocr_pipeline.types import Cell
from src.table_ocr_pipeline.utils import create_synthetic_table


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


def run_pipeline(image_path: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    pipeline = TableOCRPipeline(config)
    return pipeline.run(image_path, OUTPUT_DIR)


def fill_synthetic_text() -> None:
    cells = []
    for row_idx, row in enumerate(SYNTHETIC_TEXT):
        for col_idx, text in enumerate(row):
            cells.append(Cell(row=row_idx, col=col_idx, bbox=(0, 0, 0, 0), text=text, score=1.0))
    write_csv(SYNTHETIC_TEXT, OUTPUT_DIR / "result.csv")
    write_html(SYNTHETIC_TEXT, OUTPUT_DIR / "result.html")
    write_json(cells, OUTPUT_DIR / "result.json")


def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def execute_notebook(notebook_path: Path) -> tuple[bool, str, Path]:
    executed_dir = OUTPUT_DIR / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    output_name = notebook_path.stem + "_executed.ipynb"
    command = [
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
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    return completed.returncode == 0, (completed.stdout + "\n" + completed.stderr).strip(), executed_dir / output_name


st.set_page_config(page_title="Table OCR Demo", layout="wide")
st.title("Table OCR Pipeline Demo")
st.caption("Table Detection -> Structure Recognition -> Cell OCR -> Reconstruction")

with st.sidebar:
    st.header("Pipeline config")
    config_label = st.selectbox("Mode", list(CONFIG_OPTIONS.keys()))
    selected_config = CONFIG_OPTIONS[config_label]
    st.caption(str(selected_config.relative_to(ROOT)))

    st.header("Notebook runner")
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if notebooks:
        selected = st.selectbox("Notebook", notebooks, format_func=lambda path: path.name)
        if st.button("Execute notebook"):
            with st.spinner(f"Executing {selected.name}"):
                ok, log, executed_path = execute_notebook(selected)
            st.success("Notebook executed") if ok else st.error("Notebook execution failed")
            if executed_path.exists():
                st.download_button("Download executed notebook", read_bytes(executed_path), file_name=executed_path.name)
            if log:
                st.code(log[-4000:])
    else:
        st.info("No notebooks found.")

left, right = st.columns([0.45, 0.55])

with left:
    uploaded = st.file_uploader("Upload table image", type=["png", "jpg", "jpeg"])
    use_sample = st.button("Use sample table")
    image_path: Path | None = None
    is_sample = False

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(uploaded.getbuffer())
            image_path = Path(file.name)
    elif use_sample:
        image_path = create_synthetic_table(OUTPUT_DIR / "synthetic_table.png")
        is_sample = True

    if image_path:
        st.image(Image.open(image_path), caption="Input image", use_container_width=True)
        if st.button("Run OCR pipeline", type="primary"):
            with st.spinner("Running pipeline"):
                result = run_pipeline(image_path, selected_config)
                if is_sample:
                    fill_synthetic_text()
                st.session_state["result"] = result

with right:
    result = st.session_state.get("result")
    if not result:
        st.info("Upload an image or use the sample table, then run the pipeline.")
    else:
        st.success(f"Done: {result['tables']} table(s), {result['cells']} cell(s)")
        crop_path = OUTPUT_DIR / "table_0.png"
        if crop_path.exists():
            st.image(Image.open(crop_path), caption="Detected table crop", use_container_width=True)

        csv_path = Path(result["csv"])
        html_path = Path(result["html"])
        json_path = Path(result["json"])
        st.download_button("Download CSV", read_bytes(csv_path), file_name="table_ocr_result.csv")
        st.download_button("Download HTML", read_bytes(html_path), file_name="table_ocr_result.html")
        st.download_button("Download JSON", read_bytes(json_path), file_name="table_ocr_result.json")

        if html_path.exists():
            st.subheader("HTML preview")
            components.html(html_path.read_text(encoding="utf-8"), height=320, scrolling=True)
