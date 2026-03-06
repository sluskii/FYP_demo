from __future__ import annotations
from typing import Any, Dict, List, Tuple

import pandas as pd


def to_vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def aggregate_docs_from_hits(hits: List[Tuple]) -> pd.DataFrame:
    """
    hits rows are: (chunk_id, image_uuid, chunk_text, distance)
    """
    if not hits:
        return pd.DataFrame(columns=["image_uuid", "best_distance", "hit_count"])

    rows = [{"image_uuid": img, "distance": float(dist)} for _, img, _, dist in hits]
    df = pd.DataFrame(rows)

    agg = (
        df.groupby("image_uuid")
        .agg(best_distance=("distance", "min"), hit_count=("distance", "count"))
        .reset_index()
        .sort_values(["best_distance", "hit_count"], ascending=[True, False])
    )
    return agg


def select_chunks_for_top_docs(
    hits: List[Tuple],
    top_docs: List[str],
    per_doc: int = 3,
) -> List[Tuple]:
    """
    Pick best (lowest distance) chunks per selected doc.
    """
    by_doc: Dict[str, List[Tuple]] = {}
    for row in hits:
        chunk_id, image_uuid, chunk_text, dist = row
        by_doc.setdefault(image_uuid, []).append(row)

    selected: List[Tuple] = []
    for doc in top_docs:
        doc_hits = sorted(by_doc.get(doc, []), key=lambda r: float(r[3]))
        selected.extend(doc_hits[:per_doc])

    return selected


def build_prompt_and_source_map(user_query: str, selected_hits: List[Tuple]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (prompt, source_map)
    source_map: [{source, image_uuid, chunk_id, cosine_distance, preview}]
    """
    sources_lines = []
    source_map: List[Dict[str, Any]] = []

    for i, (chunk_id, image_uuid, chunk_text, dist) in enumerate(selected_hits, 1):
        tag = f"S{i}"
        sources_lines.append(
            f"[{tag}] image_uuid={image_uuid} chunk_id={chunk_id} cosine_distance={float(dist):.4f}\n{chunk_text}"
        )
        source_map.append(
            {
                "source": tag,
                "image_uuid": image_uuid,
                "chunk_id": chunk_id,
                "cosine_distance": float(dist),
                "preview": (chunk_text[:180] + "…") if len(chunk_text) > 180 else chunk_text,
            }
        )

    sources_block = "\n\n".join(sources_lines)

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the sources below.\n"
        "Cite sources in-line using [S1], [S2], etc.\n"
        "If the sources do not contain enough information, say you don't know.\n\n"
        f"Question: {user_query}\n\n"
        f"Sources:\n{sources_block}\n\n"
        "Answer (with citations like [S1], [S2]):"
    )
    return prompt, source_map