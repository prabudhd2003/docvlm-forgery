"""Evidence-Band Verification (EBV, called CBVR / csev / verified_recovery in the course code).

The model locates where the answer is printed; we cut a tight band and a taller
band around it, read each band separately, and accept only if both agree.
Variants differ only in who reads the taller band:

    D6  EBV-Base   one reader localizes and reads both bands
    D7  EBV-Cross  M1 localizes and reads the tight band; M3 reads the taller band
    D8  EBV-OCR    M1 localizes and reads the tight band; OCR reads the taller band (receipts only)

Also returned from the same observation (no extra model calls):
    D0  none       full-page answer
    D4  agree      full-page answer if it equals the tight-band answer, else refuse
    D5  band       tight-band answer alone

Receipt geometry and parsing are copied from scripts/colab_receipt_study.py and
src/csev/defense.py; resume geometry from scripts/colab_resume_recovery.py (policy v4).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from typing import Any, Callable

from paper.common import ROOT

sys.path.insert(0, str(ROOT / "src"))
from normalization import normalize_receipt_total, normalize_resume_degree  # noqa: E402


# ---------------- receipts ----------------------------------------------------------------
def receipt_parse_box(raw: str, frame: tuple[int, int], size: tuple[int, int], pol: dict) -> tuple | None:
    """Model box (in ``frame`` units) -> pixel box on the original image, or None."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        val = json.loads(s)["box"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(val, list) or len(val) != 4:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in val):
        return None
    fw, fh = frame
    permille = [x * 1000 / d for x, d in zip(val, (fw, fh, fw, fh))]
    if any(not 0 <= v <= 1000 for v in permille):
        return None
    w, h = size
    l, t, r, b = [round(v * d / 1000) for v, d in zip(permille, (w, h, w, h))]
    if r <= l or b <= t:
        return None
    m = pol["minimum_box_pixels"]
    if r - l < m:
        l = max(0, min(w - m, (l + r - m) // 2)); r = min(w, l + m)
    if b - t < m:
        t = max(0, min(h - m, (t + b - m) // 2)); b = min(h, t + m)
    if (r - l) * (b - t) / (w * h) > pol["maximum_box_area_fraction"]:
        return None
    return (l, t, r, b)


def receipt_band(box, size, pol):
    if box is None:
        return None
    W, H = size
    _, t, _, b = box
    hh = max(round(pol["band_min_height"] * H), b - t + 2 * round(pol["band_margin"] * H))
    if hh > pol["band_max_height"] * H:
        return None
    top = max(0, min(H - hh, round((t + b - hh) / 2)))
    return (0, top, W, top + hh)


def receipt_expand(band, size, pol):
    if band is None:
        return None
    W, H = size
    _, t, _, b = band
    pad = round(H * pol["expanded_margin"])
    top, bottom = max(0, t - pad), min(H, b + pad)
    return (0, top, W, bottom) if (bottom - top) / H <= pol["expanded_max_height"] else None


def observe_receipt(image, dataset: str, primary, wide_readers: dict[str, Any], cfg: dict,
                    ocr_extract: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    """One observation of a single-page receipt.

    ``primary`` reads the full page, localizes and reads the tight band.
    ``wide_readers`` maps a label to the reader of the taller band; the label
    "OCR" is treated as an OCR engine whose text goes through ``ocr_extract``.
    """
    pol, prompts = cfg["ebv_policy"]["receipt"], cfg["prompts"]
    norm = lambda s: normalize_receipt_total(s, dataset)  # noqa: E731
    q = prompts["receipt"]
    start = time.perf_counter()
    o: dict[str, Any] = {"calls": 0}
    o["full_raw"] = primary.read([image], q); o["calls"] += 1
    o["full"] = norm(o["full_raw"])
    o["localization_raw"] = primary.read([image], prompts["receipt_localize_relative"], pol["localization_tokens"]); o["calls"] += 1
    frame = primary.box_frame(image) if primary.box_space == "processed_pixels" else (1000, 1000)
    o["frame"] = list(frame)
    tight = receipt_parse_box(o["localization_raw"], frame, image.size, pol)
    band = receipt_band(tight, image.size, pol)
    wide = receipt_expand(band, image.size, pol)
    o.update(tight_box=tight, band=band, wide_box=wide, crop=None, crop_raw=None, wide={}, wide_raw={})
    if band is not None:
        o["crop_raw"] = primary.read([image.crop(band)], q); o["calls"] += 1
        o["crop"] = norm(o["crop_raw"])
    if wide is not None:
        for label, reader in wide_readers.items():
            if label == "OCR":
                text = reader.read([image.crop(wide)])
                o["wide_raw"][label] = text
                o["wide"][label] = norm(ocr_extract(text)) if ocr_extract else None
            else:
                o["wide_raw"][label] = reader.read([image.crop(wide)], q)
                o["wide"][label] = norm(o["wide_raw"][label])
            o["calls"] += 1
    o["seconds"] = time.perf_counter() - start
    return o


# ---------------- resumes -----------------------------------------------------------------
def resume_parse(raw: str, frame: tuple[int, int], size: tuple[int, int], pol: dict) -> list[tuple]:
    """Parse section boxes; raise ValueError on anything malformed (policy v4)."""
    s = (raw or "").strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", s, flags=re.S | re.I)
    if fences:
        if len(fences) != 1:
            raise ValueError("Ambiguous multiple JSON blocks")
        s = fences[0].strip()
    if s.lower() in {"there are none.",
                     "there are no regions indicating completed education qualifications or awarded degrees on this resume page.",
                     "there are no regions stating completed education qualifications or awarded degrees on this resume page."}:
        return []
    cleaned, quoted, escaped = [], False, False
    for i, ch in enumerate(s):
        if quoted:
            cleaned.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        else:
            if ch == '"':
                quoted = True
            if ch == "," and s[i + 1:].lstrip().startswith(("]", "}")):
                continue
            cleaned.append(ch)
    obj = json.loads("".join(cleaned))

    def unpack(x):
        if isinstance(x, dict):
            schemas = [k for k in ("boxes", "bbox_2d") if k in x]
            named = all(k in x for k in ("left", "top", "right", "bottom"))
            if len(schemas) + int(named) != 1:
                raise ValueError("Unknown or ambiguous box schema")
            return unpack(x[schemas[0]]) if schemas else [[x[k] for k in ("left", "top", "right", "bottom")]]
        if isinstance(x, list):
            if len(x) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
                return [x]
            out = []
            for item in x:
                out.extend(unpack(item))
            return out
        raise ValueError("Invalid box representation")

    vals = unpack(obj)
    if len(vals) > pol["max_raw_boxes"]:
        raise ValueError("Too many raw boxes")
    pw, ph = frame
    w, h = size
    out = []
    for v in vals:
        if len(v) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
            raise ValueError("Invalid coordinates")
        l, t, r, b = v
        if l >= r or t >= b:
            raise ValueError("Box endpoints inverted or zero area")
        if not (0 <= l and r <= pw and 0 <= t and b <= ph):
            raise ValueError("Box outside processed image")
        top, bottom = round(t * h / ph), round(b * h / ph)
        if top >= bottom:
            raise ValueError("Box collapses when mapped to original image")
        out.append((0, top, w, bottom))
    return out


def resume_views(boxes, size, pol):
    w, h = size
    intervals = []
    for _, t, _, b in boxes:
        height = max(round(pol["minimum_height"] * h), b - t + 2 * round(pol["margin"] * h))
        if height > pol["max_height"] * h:
            return None
        top = max(0, min(h - height, round((t + b - height) / 2)))
        intervals.append((top, top + height))
    merged = []
    for t, b in sorted(intervals):
        if merged and t <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((t, b))
    if len(merged) > pol["max_regions_per_page"]:
        return None
    pairs = []
    for t, b in merged:
        t2, b2 = max(0, t - round(pol["expansion"] * h)), min(h, b + round(pol["expansion"] * h))
        if b - t > pol["max_height"] * h or b2 - t2 > pol["expanded_max_height"] * h:
            return None
        pairs.append(((0, t, w, b), (0, t2, w, b2)))
    return pairs


def _resume_localize(reader, image, cfg):
    pol, prompts = cfg["ebv_policy"]["resume"], cfg["prompts"]
    if reader.box_space == "processed_pixels":
        frame = reader.box_frame(image)
        prompt = prompts["resume_localize_pixels"].replace("{w}", str(frame[0])).replace("{h}", str(frame[1]))
    else:
        frame, prompt = (1000, 1000), prompts["resume_localize_relative"]
    attempts = []
    for attempt in range(pol["localization_attempts"]):
        query = prompt
        if attempt:
            query += (" Your previous localization output was invalid. Inspect the page again and return fresh section boxes."
                      f" Every box must satisfy 0 <= left < right <= {frame[0]} and 0 <= top < bottom <= {frame[1]}."
                      " Use positive-area boxes, group nearby education lines, and output only the requested JSON. Do not guess any qualification.")
        raw = reader.read([image], query, pol["localization_tokens"])
        try:
            pairs = resume_views(resume_parse(raw, frame, image.size, pol), image.size, pol)
            if pairs is None:
                raise ValueError("Merged regions exceed size or count limits")
            attempts.append({"raw": raw, "status": "valid"})
            return {"status": "ok" if pairs else "no_education", "pairs": pairs, "attempts": attempts, "frame": list(frame)}
        except (ValueError, KeyError, TypeError) as exc:
            attempts.append({"raw": raw, "status": "invalid", "error": str(exc)})
    return {"status": "invalid", "pairs": [], "attempts": attempts, "frame": list(frame)}


def observe_resume(pages, primary, wide_readers: dict[str, Any], cfg: dict) -> dict[str, Any]:
    pol, q = cfg["ebv_policy"]["resume"], cfg["prompts"]["resume"]
    start = time.perf_counter()
    o: dict[str, Any] = {"calls": 0}
    o["full_raw"] = primary.read(pages, q); o["calls"] += 1
    o["full"] = normalize_resume_degree(o["full_raw"])
    regions, bands, expanded, valid = [], [], [], True
    for j, im in enumerate(pages):
        r = _resume_localize(primary, im, cfg); r["page"] = j; regions.append(r)
        o["calls"] += len(r["attempts"])
        if r["status"] == "invalid":
            valid = False
            continue
        for b, e in r["pairs"]:
            bands.append(im.crop(b)); expanded.append(im.crop(e))
    o.update(regions=regions, crop=None, crop_raw=None, wide={}, wide_raw={}, band_count=len(bands))
    if valid and 0 < len(bands) <= pol["max_crops_per_document"]:
        o["crop_raw"] = primary.read(bands, q); o["calls"] += 1
        o["crop"] = normalize_resume_degree(o["crop_raw"])
        for label, reader in wide_readers.items():
            if label == "OCR":
                continue  # no OCR variant for resumes
            o["wide_raw"][label] = reader.read(expanded, q); o["calls"] += 1
            o["wide"][label] = normalize_resume_degree(o["wide_raw"][label])
    o["seconds"] = time.perf_counter() - start
    return o


# ---------------- decisions ---------------------------------------------------------------
def decisions(o: dict[str, Any], base_label: str, cross_label: str | None = "M3", ocr_label: str | None = "OCR") -> dict[str, str | None]:
    """Map one observation to every method's answer (None = refused / sent to a person)."""
    full, crop, wide = o["full"], o["crop"], o["wide"]
    agree = lambda a, b: a if a is not None and a == b else None  # noqa: E731
    out = {"D0": full, "D4": agree(full, crop), "D5": crop, "D6": agree(crop, wide.get(base_label))}
    if cross_label:
        out["D7"] = agree(crop, wide.get(cross_label))
    if ocr_label:
        out["D8"] = agree(crop, wide.get(ocr_label))
    return out


def outcome(pred: str | None, clean: str, target: str | None) -> dict[str, Any]:
    return {"prediction": pred, "accepted": pred is not None, "correct": pred is not None and pred == clean,
            "targeted": pred is not None and target is not None and pred == target}
