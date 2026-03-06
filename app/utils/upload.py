from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List

import streamlit as st

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    name = (name or "upload").strip()
    name = SAFE_NAME_RE.sub("_", name)
    return name[:180]


def save_uploaded_image(uploaded_file, upload_dir: str, overwrite: bool = False) -> str:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise ValueError(f"Unsupported file type: {ext}")

    base = safe_filename(Path(uploaded_file.name).stem)
    filename = f"{base}{ext}"

    out_path = Path(upload_dir) / filename

    if out_path.exists() and not overwrite:
        filename = f"{base}_{int(time.time())}{ext}"
        out_path = Path(upload_dir) / filename

    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return filename  # return filename only (not full path)


def upload_sidebar(upload_dir: str) -> List[str]:
    """
    Sidebar upload section.
    Upload directory is defined in backend only.
    Returns list of saved filenames.
    """

    os.makedirs(upload_dir, exist_ok=True)

    st.divider()
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload document images (.png/.jpg/.jpeg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    overwrite = st.checkbox("Overwrite if filename exists", value=False)

    saved_files: List[str] = []

    if uploaded_files:
        if st.button("Save uploaded files"):
            failed = []

            for uf in uploaded_files:
                try:
                    filename = save_uploaded_image(uf, upload_dir, overwrite)
                    saved_files.append(filename)
                except Exception as e:
                    failed.append((uf.name, str(e)))

            if saved_files:
                st.success(f"✅ Saved {len(saved_files)} file(s)")
                for f in saved_files:
                    st.write(f"- {f}")

            if failed:
                st.error("Some files failed:")
                for name, err in failed:
                    st.write(f"- {name}: {err}")

    return saved_files