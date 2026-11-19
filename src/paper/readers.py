"""Inference-only document readers used in paper_v1 (no gradients anywhere).

    python -m paper.readers pin              # print current Hugging Face revisions for configs/paper_v1.json
    python -m paper.readers smoke M2         # load one reader and read 3 R-dev receipts

Every reader exposes:
    read(pages, prompt, max_new_tokens=None) -> str      greedy, batch size 1
    box_frame(image) -> (w, h)                           coordinate frame the model answers boxes in
    settings -> dict                                     recorded with every result
    unload()
"""
from __future__ import annotations

import gc
import re
import sys
from typing import Any

from paper.common import ROOT, load_config

sys.path.insert(0, str(ROOT / "src"))


def _torch_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    return torch.device("cuda")


def _check_revision(name: str, spec: dict) -> None:
    import os

    if spec["revision"] == "PIN_IN_E0" and os.environ.get("PAPER_ALLOW_UNPINNED") != "1":
        raise RuntimeError(f"{name} has no pinned revision. Run `python -m paper.readers pin`, put the hash in configs/paper_v1.json, commit.")


def _free():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class QwenReader:
    """Qwen2.5-VL via the course wrapper (identical clean path to the course results)."""

    def __init__(self, name, spec, gen):
        from models.qwen import QwenVL

        self.name, self.box_space = name, spec["box_space"]
        revision = None if spec["revision"] == "PIN_IN_E0" else spec["revision"]
        self.w = QwenVL(device=_torch_device(), model_id=spec["model_id"], revision=revision,
                        precision=gen["precision"], min_pixels=gen["min_pixels"], max_pixels=gen["max_pixels"],
                        max_new_tokens=gen["max_new_tokens"])
        self.settings = {"model_id": spec["model_id"], "revision": spec["revision"], **self.w.settings}

    def _clear(self):
        model = self.w.model
        for item in (model, *model.modules()):
            if hasattr(item, "rope_deltas"):
                item.rope_deltas = None

    def read(self, pages, prompt, max_new_tokens=None):
        self._clear()
        old = self.w.max_new_tokens
        try:
            if max_new_tokens:
                self.w.max_new_tokens = int(max_new_tokens)
            return self.w.predict(list(pages), prompt)
        finally:
            self.w.max_new_tokens = old

    def box_frame(self, image):
        ip = self.w.processor.image_processor
        grid = ip(images=[image], return_tensors="pt")["image_grid_thw"].tolist()[0]
        return (int(grid[2]) * int(ip.patch_size), int(grid[1]) * int(ip.patch_size))

    def unload(self):
        self.w.unload()
        _free()


class HFChatReader:
    """Generic chat VLM (InternVL3.5-HF, Gemma-3, Qwen3-VL) through AutoModelForImageTextToText."""

    def __init__(self, name, spec, gen):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name, self.box_space = name, spec["box_space"]
        self.max_new_tokens = gen["max_new_tokens"]
        revision = None if spec["revision"] == "PIN_IN_E0" else spec["revision"]
        dtype = torch.bfloat16 if gen["precision"] == "bfloat16" else torch.float16
        self.device = _torch_device()
        self.processor = AutoProcessor.from_pretrained(spec["model_id"], revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(spec["model_id"], revision=revision, torch_dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.settings = {"model_id": spec["model_id"], "revision": spec["revision"], "precision": gen["precision"],
                         "max_new_tokens": self.max_new_tokens, "do_sample": False, "batch_size": 1,
                         "image_processing": "model default"}

    def read(self, pages, prompt, max_new_tokens=None):
        import torch

        content = [{"type": "image", "image": p.convert("RGB")} for p in pages] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens or self.max_new_tokens), do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def box_frame(self, image):
        return (1000, 1000)

    def unload(self):
        del self.model, self.processor
        _free()


class DonutReader:
    box_space = "none"

    def __init__(self, name, spec, gen):
        from models.donut import DonutDocVQA

        self.name = name
        self.w = DonutDocVQA(device=_torch_device(), model_id=spec["model_id"], revision=spec["revision"])
        self.settings = {"model_id": spec["model_id"], "revision": spec["revision"], "max_length": self.w.max_length}

    def read(self, pages, prompt, max_new_tokens=None):
        return self.w.predict(list(pages)[:1], prompt)

    def box_frame(self, image):
        raise NotImplementedError("Donut does not localize")

    def unload(self):
        self.w.unload()
        _free()


class OCRReader:
    """PaddleOCR on a cropped band. ``read`` returns the recognized lines, top to bottom."""

    box_space = "none"

    def __init__(self, name, spec, gen):
        from paddleocr import PaddleOCR

        self.name = name
        try:  # PaddleOCR 3.x
            self.ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            self.v3 = True
        except TypeError:  # PaddleOCR 2.x
            self.ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
            self.v3 = False
        self.settings = {"engine": "paddleocr", "revision": spec["revision"], "api": "3.x" if self.v3 else "2.x"}

    def lines(self, image) -> list[str]:
        import numpy as np

        array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        items: list[tuple[float, str]] = []
        if self.v3:
            for res in self.ocr.predict(array):
                for text, poly in zip(res["rec_texts"], res["rec_polys"]):
                    items.append((float(np.asarray(poly)[:, 1].mean()), str(text)))
        else:
            for line in self.ocr.ocr(array, cls=False)[0] or []:
                poly, (text, _conf) = line
                items.append((float(np.asarray(poly)[:, 1].mean()), str(text)))
        return [t for _, t in sorted(items)]

    def read(self, pages, prompt=None, max_new_tokens=None):
        return "\n".join(line for p in pages for line in self.lines(p))

    def box_frame(self, image):
        raise NotImplementedError("OCR does not localize")

    def unload(self):
        del self.ocr
        _free()


KINDS = {"qwen25": QwenReader, "hf_chat": HFChatReader, "donut": DonutReader, "paddleocr": OCRReader}


def load_reader(name: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    spec = cfg["readers"][name]
    _check_revision(name, spec)
    return KINDS[spec["kind"]](name, spec, cfg["generation"])


# ---- OCR answer extraction (label-free, deterministic) ---------------------------------
_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}\b|\d{1,3}(?:[.,]\d{3})+\b|\d+\b")
_TOTAL_WORDS = ("grand total", "total", "amount due", "balance due", "amount paid", "jumlah", "tunai", "total bayar")
_EXCLUDE = ("subtotal", "sub total", "sub-total", "tax", "change", "kembali", "cash", "discount")


def ocr_receipt_total(text: str | None) -> str | None:
    """Pick the amount on the one line that names the total; else the only amount in the band.

    Returns the raw amount string (normalize it with the dataset rule), or None to refuse.
    """
    if not text:
        return None
    lines = [l for l in text.splitlines() if l.strip()]
    tagged = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(w in low for w in _TOTAL_WORDS) and not any(w in low for w in _EXCLUDE):
            amounts = _AMOUNT.findall(line)
            if not amounts and i + 1 < len(lines):  # label and value on separate lines
                amounts = _AMOUNT.findall(lines[i + 1])
            if amounts:
                tagged.append(amounts[-1])
    if len(set(tagged)) == 1:
        return tagged[0]
    if tagged:
        return None
    every = [a for line in lines for a in _AMOUNT.findall(line)]
    return every[0] if len(set(every)) == 1 else None


# ---- CLI --------------------------------------------------------------------------------
def _pin(cfg):
    from huggingface_hub import model_info

    for name, spec in cfg["readers"].items():
        if spec["kind"] == "paddleocr":
            continue
        info = model_info(spec["model_id"])
        flag = "" if spec["revision"] == info.sha else "   <- update configs/paper_v1.json"
        print(f"{name:4s} {spec['model_id']:45s} {info.sha}{flag}")
    try:
        import importlib.metadata as md

        print(f"OCR  paddleocr {md.version('paddleocr')}  (put this version string as OCR revision)")
    except Exception:
        print("OCR  paddleocr not installed: pip install paddleocr paddlepaddle")


def _smoke(name, cfg):
    import time

    import torch

    from paper import documents, sets
    from normalization import normalize_receipt_total

    ids = sets.get("R-dev")[:3]
    pages = documents.load_pages(ids)
    torch.cuda.reset_peak_memory_stats()
    reader = load_reader(name, cfg)
    for rid in ids:
        t = time.perf_counter()
        raw = reader.read(pages[rid], cfg["prompts"]["receipt"]) if name != "OCR" else reader.read(pages[rid])
        m = documents.meta(rid)
        pred = normalize_receipt_total(raw if name != "OCR" else ocr_receipt_total(raw), m["dataset_name"])
        print(f"{rid}: raw={raw!r} pred={pred} truth={m['clean_target']} {time.perf_counter() - t:.1f}s")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; settings: {reader.settings}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pin", "smoke"])
    p.add_argument("name", nargs="?")
    a = p.parse_args()
    cfg = load_config()
    if a.command == "pin":
        _pin(cfg)
    else:
        _smoke(a.name, cfg)


if __name__ == "__main__":
    main()
