"""Document metadata and clean images for every record ID used in paper_v1.

Receipts from our own collection are read from ``data/collected_receipts`` with
the same EXIF transpose + RGB conversion as ``scripts/freeze_collected_external.py``.
Their IDs ``collected_receipts/external_test/cNNNNN`` follow that script: the
index of the image filename among verified labels sorted by filename.
Public documents come from ``data/processed/<dataset>`` (HF ``save_to_disk``).
"""
from __future__ import annotations

import csv
import functools
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from paper.common import ROOT

COLLECTED_PREFIX = "collected_receipts/external_test/"
LABELS = ROOT / "data" / "annotations" / "collected_receipt_labels.csv"
COLLECTED_IMAGES = ROOT / "data" / "collected_receipts"
MASTER_MANIFEST = ROOT / "data" / "processed" / "master_manifest.csv"


def _money(value: str) -> str:
    amount = Decimal(value.replace("$", "").replace(",", "").strip())
    return f"{amount.quantize(Decimal('0.01')):.2f}"


@functools.lru_cache(maxsize=1)
def collected_index() -> dict[str, dict[str, Any]]:
    with LABELS.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status", "").lower() == "verified"]
    rows.sort(key=lambda r: r["image_filename"])
    out = {}
    for i, r in enumerate(rows):
        clean = _money(r["visible_total"])
        rid = f"{COLLECTED_PREFIX}c{i:05d}"
        out[rid] = {
            "record_id": rid, "dataset_name": "collected_receipts", "task": "receipt_total",
            "experiment_split": "external_test", "image_filename": r["image_filename"],
            "clean_target": clean,
            "attack_target": f"{(Decimal(clean) * 10).quantize(Decimal('0.01')):.2f}",
            "num_pages": 1,
        }
    return out


@functools.lru_cache(maxsize=1)
def public_index() -> dict[str, dict[str, Any]]:
    out = {}
    with MASTER_MANIFEST.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["usable"].strip().lower() != "true":
                continue
            out[r["record_id"]] = {
                "record_id": r["record_id"], "dataset_name": r["dataset_name"], "task": r["task"],
                "experiment_split": r["experiment_split"], "source_split": r["source_split"],
                "clean_target": r["clean_target"], "attack_target": r["attack_target"],
                "num_pages": int(float(r["num_pages"] or 1)),
            }
    return out


def meta(record_id: str) -> dict[str, Any]:
    if record_id.startswith(COLLECTED_PREFIX):
        return collected_index()[record_id]
    return public_index()[record_id]


def task_of(record_id: str) -> str:
    return "receipt" if meta(record_id)["task"] == "receipt_total" else "resume"


def load_pages(record_ids: list[str]) -> dict[str, list]:
    """Return {record_id: [PIL RGB pages]} for clean documents."""
    from PIL import Image, ImageOps

    out: dict[str, list] = {}
    public: dict[str, list[str]] = {}
    for rid in record_ids:
        if rid.startswith(COLLECTED_PREFIX):
            m = collected_index()[rid]
            with Image.open(COLLECTED_IMAGES / m["image_filename"]) as im:
                out[rid] = [ImageOps.exif_transpose(im).convert("RGB")]
        else:
            public.setdefault(public_index()[rid]["experiment_split"], []).append(rid)
    if public:
        sys.path.insert(0, str(ROOT / "src"))
        from data import load_processed_rows, row_images

        for split, ids in public.items():
            rows = load_processed_rows(ROOT, split, ids)
            for rid in ids:
                out[rid] = [p.convert("RGB") for p in row_images(rows[rid])]
    return out


def load_image_file(path: str | Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB").copy()
