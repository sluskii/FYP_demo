from __future__ import annotations
import os
from typing import Optional


def find_image_path(image_uuid: str, base_dir: str, extra_dirs: Optional[list[str]] = None) -> Optional[str]:
    """
    Try to find image_uuid.{png|jpg|jpeg} in base_dir (and optional extra_dirs).
    """
    if not image_uuid:
        return None

    search_dirs = [base_dir]
    if extra_dirs:
        search_dirs.extend([d for d in extra_dirs if d])

    candidates = [
        f"{image_uuid}.png", f"{image_uuid}.jpg", f"{image_uuid}.jpeg",
        f"{image_uuid}.PNG", f"{image_uuid}.JPG", f"{image_uuid}.JPEG",
    ]

    for d in search_dirs:
        for fname in candidates:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p

    return None