"""Evaluation experiments: how units are listed and how one unit is executed.

    E1     clean full-page accuracy, 6 readers x (R-test + Res-test)
    E4dev  defenses D0-D8 on R-dev + Res-dev, clean (+ course attacks via --manifest course_rdev); tuning only
    E4     defenses D0-D8 (M1 primary) on R-test + Res-test, clean + every E2 attack     [needs frozen policy]
    E4r    EBV-Base (D0, D4, D5, D6) with M2 and with M3 as the reader, clean + E2a attacks
    E3     every E2/E3b attacked image read full-page by all 6 readers (attack success + transfer)
    E5     final images of direct attacks (E5a/E5b/E5c/E6) under D0-D8                  [needs frozen policy]
    E7     target-ablation images read full-page by M1
    E8     stealth (SSIM, PSNR, LPIPS) for every attacked image; no GPU model needed
    E8lat  seconds and peak memory per document for D0, D6, D7, D8 on 50 clean R-test receipts
    E9     successful M1 eps 8/16 attacks after JPEG q75/q85 and downscale to 1600 px, read by D0 and D6
"""
from __future__ import annotations

import io
import time
from typing import Any

from paper import documents, ebv, sets
from paper.common import digest, environment
from paper.readers import ocr_receipt_total

FULL_READERS = ["M1", "M2", "M3", "T1", "T2", "T3"]
NEEDS_FROZEN_POLICY = {"E4", "E4r", "E5", "E8lat", "E9"}


def _doc_unit(exp, record_id, **extra):
    u = {"experiment": exp, "kind": "clean", "record_id": record_id, "attack_id": None, "family": "clean",
         "epsilon": 0, "source": None, "target": documents.meta(record_id)["attack_target"], **extra}
    u["unit_id"] = digest([exp, record_id, extra.get("reader"), extra.get("transform"), extra.get("method")])[:24]
    return u


def _attack_unit(exp, a, **extra):
    u = {"experiment": exp, "kind": "attack", "record_id": a["record_id"], "attack_id": a["attack_id"],
         "family": a["family"], "epsilon": a["epsilon"], "source": a["source"], "target": a["target"],
         "target_kind": a.get("target_kind", "x10"), "attacked_method": a.get("attacked_method"),
         "locator_goal": a.get("locator_goal"),
         "restart": a.get("restart", 0), "steps": a.get("steps"), "attack_experiment": a["experiment"],
         "images": a["images"], **extra}
    u["unit_id"] = digest([exp, a["attack_id"], extra.get("reader"), extra.get("transform"), extra.get("method")])[:24]
    return u


def units(exp: str, cfg: dict, attacks: list[dict]) -> list[dict]:
    """List every unit; grouped by the readers they need so one task loads models once."""
    is_receipt = lambda rid: documents.task_of(rid) == "receipt"  # noqa: E731
    if exp == "E1":
        docs = sets.get("R-test") + sets.get("Res-test")
        return [_doc_unit(exp, r, reader=m) for m in FULL_READERS for r in docs if m != "T3" or is_receipt(r)]
    if exp == "E4dev":
        return [_doc_unit(exp, r) for r in sets.get("R-dev") + sets.get("Res-dev")] + [_attack_unit(exp, a) for a in attacks]
    if exp == "E4":
        return [_doc_unit(exp, r) for r in sets.get("R-test") + sets.get("Res-test")] + [_attack_unit(exp, a) for a in attacks]
    if exp == "E4r":
        docs = sets.get("R-test") + sets.get("Res-test")
        return [u for m in ("M2", "M3") for u in [_doc_unit(exp, r, reader=m) for r in docs] + [_attack_unit(exp, a, reader=m) for a in attacks]]
    if exp == "E3":
        return [_attack_unit(exp, a, reader=m) for m in FULL_READERS for a in attacks if m != "T3" or is_receipt(a["record_id"])]
    if exp == "E5":
        return [_attack_unit(exp, a) for a in attacks]
    if exp == "E7":
        return [_attack_unit(exp, a, reader="M1") for a in attacks]
    if exp == "E8":
        return [_attack_unit(exp, a) for a in attacks]
    if exp == "E8lat":
        docs = [r for r in sets.get("R-test") if is_receipt(r)][:50]
        return [_doc_unit(exp, r, method=m) for m in ("D0", "D6", "D7", "D8") for r in docs]
    if exp == "E9":
        return [_attack_unit(exp, a, transform=t) for t in cfg["upload_transforms"] for a in attacks]
    raise SystemExit(f"unknown experiment {exp}")


def readers_needed(u: dict) -> tuple[str, ...]:
    exp = u["experiment"]
    if exp in ("E1", "E3", "E7", "E4r"):
        return (u["reader"],)
    if exp == "E8":
        return ()
    receipt = documents.task_of(u["record_id"]) == "receipt"
    if exp == "E8lat":
        return {"D0": ("M1",), "D6": ("M1",), "D7": ("M1", "M3"), "D8": ("M1", "OCR")}[u["method"]]
    if exp == "E9":
        return ("M1",)
    return ("M1", "M3", "OCR") if receipt else ("M1", "M3")


# ---------------- execution ---------------------------------------------------------------
def _pages(u: dict, clean_pages: dict) -> list:
    if u["kind"] == "clean":
        return clean_pages[u["record_id"]]
    from paper.manifest import _abs

    return [documents.load_image_file(_abs(i["path"])) for i in u["images"]]


def _normalizer(record_id):
    from normalization import normalize_receipt_total, normalize_resume_degree

    m = documents.meta(record_id)
    if m["task"] == "receipt_total":
        return lambda s: normalize_receipt_total(s, m["dataset_name"])
    return normalize_resume_degree


def _prompt(cfg, record_id):
    return cfg["prompts"]["receipt" if documents.task_of(record_id) == "receipt" else "resume"]


def _baselines(pages, u, reader, cfg):
    from defenses import evaluate_defense

    norm, q = _normalizer(u["record_id"]), _prompt(cfg, u["record_id"])
    out = {}
    for name, key in (("jpeg", "D1"), ("resize_vote", "D2"), ("selective_multiview", "D3")):
        res, _ = evaluate_defense(pages=pages, defense_id=name, record_id=u["record_id"],
                                  predict=lambda p: reader.read(p, q), normalize=norm, config=cfg["baselines"])
        out[key] = None if res.abstained else res.final_prediction
    return out


def _observe(pages, u, primary_label, wide_labels, R, cfg):
    m = documents.meta(u["record_id"])
    wides = {k: R[k] for k in wide_labels}
    if m["task"] == "receipt_total":
        if len(pages) != 1:
            raise ValueError("receipts must have one page")
        return ebv.observe_receipt(pages[0], m["dataset_name"], R[primary_label], wides, cfg, ocr_receipt_total)
    return ebv.observe_resume(pages, R[primary_label], {k: v for k, v in wides.items() if k != "OCR"}, cfg)


def upload_transform(image, name: str):
    from PIL import Image

    if name.startswith("jpeg_q"):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(name[6:]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name.startswith("downscale_"):
        side = int(name.split("_")[1])
        scale = side / max(image.size)
        if scale >= 1:
            return image.copy()
        return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    raise ValueError(name)


def stealth(clean, attacked, lpips_model=None) -> dict[str, float]:
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a = np.asarray(clean, dtype=np.float32) / 255
    b = np.asarray(attacked, dtype=np.float32) / 255
    out = {"ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
           "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
           "linf_255": int(np.abs(np.rint(a * 255) - np.rint(b * 255)).max())}
    if lpips_model is not None:
        import torch

        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].mul(2).sub(1).to(next(lpips_model.parameters()).device)  # noqa: E731
        with torch.inference_mode():
            out["lpips"] = float(lpips_model(t(a), t(b)).item())
    return out


def execute(u: dict, R: dict, cfg: dict, clean_pages: dict, extras: dict) -> dict[str, Any]:
    """Run one unit. ``R`` holds loaded readers; ``extras`` holds e.g. the LPIPS model."""
    exp, rid = u["experiment"], u["record_id"]
    meta = documents.meta(rid)
    clean, target = meta["clean_target"], u["target"]
    start = time.perf_counter()
    rec: dict[str, Any] = {"unit": u, "clean_target": clean}
    if exp in ("E1", "E3", "E7"):
        pages = _pages(u, clean_pages)
        raw = R[u["reader"]].read(pages[:1] if u["reader"] == "T3" else pages, _prompt(cfg, rid))
        rec["raw"] = raw
        rec["methods"] = {"D0": ebv.outcome(_normalizer(rid)(raw), clean, target)}
    elif exp == "E8":
        pages = _pages(u, clean_pages)
        rec["pages"] = [stealth(c, a, extras.get("lpips")) for c, a in zip(clean_pages[rid], pages)]
    elif exp == "E8lat":
        import torch

        pages = clean_pages[rid]
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        if u["method"] == "D0":
            R["M1"].read(pages, _prompt(cfg, rid))
        else:
            wide = {"D6": ["M1"], "D7": ["M3"], "D8": ["OCR"]}[u["method"]]
            _observe(pages, u, "M1", wide, R, cfg)
        rec["latency"] = {"seconds": time.perf_counter() - t, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    else:
        pages = _pages(u, clean_pages)
        if exp == "E9":
            pages = [upload_transform(p, u["transform"]) for p in pages]
        if exp == "E4r":
            o = _observe(pages, u, u["reader"], [u["reader"]], R, cfg)
            dec = ebv.decisions(o, u["reader"], None, None)
        elif exp == "E9":
            o = _observe(pages, u, "M1", ["M1"], R, cfg)
            dec = {k: v for k, v in ebv.decisions(o, "M1", None, None).items() if k in ("D0", "D6")}
        else:
            receipt = meta["task"] == "receipt_total"
            o = _observe(pages, u, "M1", ["M1", "M3", "OCR"] if receipt else ["M1", "M3"], R, cfg)
            dec = ebv.decisions(o, "M1", "M3", "OCR" if receipt else None)
            dec.update(_baselines(pages, u, R["M1"], cfg))
        rec["observation"] = o
        rec["methods"] = {k: ebv.outcome(v, clean, target) for k, v in sorted(dec.items())}
    rec["seconds"] = time.perf_counter() - start
    rec["environment"] = extras.get("environment") or environment()
    return rec
