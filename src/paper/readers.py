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


class TesseractReader:
    """Tesseract on a cropped band, via subprocess (no CUDA, nothing shared with torch).

    Non-neural and non-differentiable, so it is the independent second reader for EBV-OCR
    (D8). Uses the same binary and tessdata the course OCR baseline used; ``read`` returns
    the recognized text, which ``ocr_receipt_total`` then turns into an amount.
    """

    box_space = "none"

    def __init__(self, name, spec, gen):
        import shutil
        from pathlib import Path

        self.name = name
        self.exe = spec.get("executable")
        if not self.exe or not Path(self.exe).exists():
            self.exe = shutil.which("tesseract") or self.exe
        if not self.exe or not Path(self.exe).exists():
            raise RuntimeError(
                "tesseract not found. Set readers.OCR.executable in configs/paper_v1.json, "
                "or make it available (module load tesseract / conda install -c conda-forge tesseract)."
            )
        self.tessdata = spec.get("tessdata_dir")
        self.lang = spec.get("language", "eng")
        self.psm = str(spec.get("psm", 3))
        self.settings = {"engine": "tesseract", "executable": self.exe, "language": self.lang,
                         "psm": self.psm, "revision": spec.get("revision")}

    def read(self, pages, prompt=None, max_new_tokens=None):
        import subprocess
        import tempfile

        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                page.convert("RGB").save(tmp.name)
                cmd = [self.exe, tmp.name, "stdout"]
                if self.tessdata:
                    cmd += ["--tessdata-dir", self.tessdata]
                cmd += ["-l", self.lang, "--oem", "1", "--psm", self.psm]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out.returncode != 0:
                raise RuntimeError(f"tesseract failed ({out.returncode}): {out.stderr.strip()[:200]}")
            texts.append(out.stdout)
        return "\n".join(texts)

    def box_frame(self, image):
        raise NotImplementedError("OCR does not localize")

    def unload(self):
        pass


def _rows_text(items) -> str:
    """Words whose vertical centre is within half the median word height of a row's first
    word share that row; rows top to bottom, words left to right."""
    import statistics

    if not items:
        return ""
    h = statistics.median(float(i[2]) for i in items) or 1.0
    items = sorted(items, key=lambda i: i[0])
    rows = [[items[0]]]
    for it in items[1:]:
        if abs(it[0] - rows[-1][0][0]) <= 0.5 * h:
            rows[-1].append(it)
        else:
            rows.append([it])
    return "\n".join(" ".join(str(t) for *_, t in sorted(r, key=lambda i: i[1])) for r in rows)


class PaddleReader:
    """PaddleOCR (CPU) on a cropped band, in a separate process and conda env.

    Paddle never shares a process or CUDA with torch. Words are grouped into rows so a
    label and its amount stay on one line for ``ocr_receipt_total``. Non-differentiable;
    the D8 gradient term uses the TrOCR stand-in, success is always checked here.
    """

    box_space = "none"

    def __init__(self, name, spec, gen):
        import json
        import os
        import subprocess
        from pathlib import Path

        self.name, self.proc = name, None
        python = Path(os.path.expanduser(spec["python"]))
        if not python.exists():
            raise RuntimeError(f"PaddleOCR python not found: {python} (create env paperwork-paddle, see env/paddle-requirements.lock.txt)")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(OMP_NUM_THREADS=str(spec.get("threads", 4)), FLAGS_use_mkldnn="0",
                   CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        self.proc = subprocess.Popen([str(python), "-u", str(ROOT / "src" / "paper" / "paddle_worker.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        hello = json.loads(self._line())
        want = {"paddleocr": spec["paddleocr"], "paddlepaddle": spec["paddlepaddle"]}
        got = {k: hello.get(k) for k in want}
        if got != want:
            self.unload()
            raise RuntimeError(f"PaddleOCR versions {got} differ from the pinned {want}")
        self.settings = {"engine": "paddleocr", "python": str(python), **got, "layout": "rows",
                         "threads": spec.get("threads", 4), "revision": spec["revision"]}

    def _line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"PaddleOCR worker exited (code {self.proc.poll()}); see stderr in the job log")
        return line

    def read(self, pages, prompt=None, max_new_tokens=None):
        import json
        import tempfile

        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                page.convert("RGB").save(tmp.name)
                self.proc.stdin.write(json.dumps({"path": tmp.name}) + "\n")
                self.proc.stdin.flush()
                reply = json.loads(self._line())
            if "error" in reply:
                raise RuntimeError(f"PaddleOCR failed: {reply['error'][:300]}")
            texts.append(_rows_text(reply["items"]))
        return "\n".join(texts)

    def box_frame(self, image):
        raise NotImplementedError("OCR does not localize")

    def unload(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


KINDS = {"qwen25": QwenReader, "hf_chat": HFChatReader, "donut": DonutReader,
         "tesseract": TesseractReader, "paddle": PaddleReader}


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
            if not amounts and i + 1 < len(lines):
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
        if spec["kind"] in ("tesseract", "paddle"):
            continue
        info = model_info(spec["model_id"])
        flag = "" if spec["revision"] == info.sha else "   <- update configs/paper_v1.json"
        print(f"{name:4s} {spec['model_id']:45s} {info.sha}{flag}")
    spec = cfg["readers"].get("OCR")
    if spec and spec["kind"] == "tesseract":
        import shutil
        import subprocess
        from pathlib import Path

        exe = spec.get("executable") if spec.get("executable") and Path(spec["executable"]).exists() else shutil.which("tesseract")
        if exe:
            try:
                ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stderr.splitlines()[0]
            except Exception:
                ver = "version unknown"
            print(f"OCR  tesseract at {exe}  ({ver})")
        else:
            print("OCR  tesseract not found; set readers.OCR.executable or `module load tesseract`")


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
