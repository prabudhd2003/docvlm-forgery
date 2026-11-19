"""CPU tests for src/paper (no GPU, no model weights). Run: python -m pytest tests/test_paper.py -q"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper import common, ebv, readers  # noqa: E402

CFG = common.load_config()


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- EBV geometry matches the course code exactly ----------------------------------
def test_receipt_band_matches_course():
    old = _load_script("colab_receipt_study")
    pol = CFG["ebv_policy"]["receipt"]
    rng = random.Random(0)
    for _ in range(2000):
        W, H = rng.randint(200, 1500), rng.randint(300, 3000)
        t = rng.randint(0, H - 2); b = rng.randint(t + 1, H)
        box = (0, t, W, b)
        assert ebv.receipt_band(box, (W, H), pol) == old.band(box, (W, H))
        band = ebv.receipt_band(box, (W, H), pol)
        assert ebv.receipt_expand(band, (W, H), pol) == old.expand(band, (W, H))


def test_receipt_parse_box_matches_course():
    sys.path.insert(0, str(ROOT / "src"))
    from csev.defense import parse_box

    settings = {"minimum_box_pixels": 28, "maximum_box_area_fraction": 0.35}
    pol = CFG["ebv_policy"]["receipt"]
    rng = random.Random(1)
    for _ in range(1000):
        size = (rng.randint(300, 1200), rng.randint(500, 2500))
        frame = (rng.randint(10, 60) * 28, rng.randint(10, 60) * 28)
        val = [rng.uniform(0, frame[0]), rng.uniform(0, frame[1]), rng.uniform(0, frame[0]), rng.uniform(0, frame[1])]
        raw = json.dumps({"box": val})
        expected = parse_box(json.dumps({"box": [x * 1000 / d for x, d in zip(val, (*frame, *frame))]}), size, settings)
        assert ebv.receipt_parse_box(raw, frame, size, pol) == expected
    assert ebv.receipt_parse_box('{"box": null}', (100, 100), (100, 100), pol) is None
    assert ebv.receipt_parse_box("not json", (100, 100), (100, 100), pol) is None


def test_resume_geometry_matches_course():
    old = _load_script("colab_resume_recovery")
    pol = CFG["ebv_policy"]["resume"]
    rng = random.Random(2)
    for _ in range(500):
        frame = (rng.randint(20, 40) * 28, rng.randint(30, 60) * 28)
        size = (rng.randint(800, 1700), rng.randint(1000, 2400))
        boxes = []
        for _ in range(rng.randint(0, 5)):
            t = rng.uniform(0, frame[1] - 20)
            boxes.append([rng.uniform(0, 50), t, rng.uniform(60, frame[0]), rng.uniform(t + 5, min(frame[1], t + 400))])
        raw = json.dumps({"boxes": boxes})
        try:
            expected = old.views(old.parse(raw, frame, size), size)
            ok = True
        except ValueError:
            ok = False
        if ok:
            assert ebv.resume_views(ebv.resume_parse(raw, frame, size, pol), size, pol) == expected
        else:
            with pytest.raises(ValueError):
                ebv.resume_views(ebv.resume_parse(raw, frame, size, pol), size, pol)


# ---------- EBV decisions with scripted readers -------------------------------------------
class Fake:
    """Answers by what it is shown: full page (height 400), tight band (52), taller band (76)."""

    box_space = "relative_1000"

    def __init__(self, full, tight, wide, loc='{"box":[100,500,900,560]}'):
        self.a = {"full": full, "tight": tight, "wide": wide, "loc": loc}
        self.calls = []

    def read(self, pages, prompt, max_new_tokens=None):
        h = pages[0].height
        kind = "loc" if "Locate" in prompt else {400: "full", 52: "tight", 76: "wide"}[h]
        self.calls.append(kind)
        return self.a[kind]

    def box_frame(self, image):
        return (1000, 1000)


class FakeOCR:
    def __init__(self, text):
        self.text = text

    def read(self, pages, prompt=None, max_new_tokens=None):
        return self.text


IMG = Image.new("RGB", (100, 400), "white")


def _obs(primary, wides):
    return ebv.observe_receipt(IMG, "sroie", primary, wides, CFG, readers.ocr_receipt_total)


def test_ebv_accepts_when_bands_agree():
    m1 = Fake("12.50", "12.50", "12.50")
    o = _obs(m1, {"M1": m1, "M3": Fake("x", "x", "12.50"), "OCR": FakeOCR("TOTAL 12.50")})
    d = ebv.decisions(o, "M1")
    assert d == {"D0": "12.50", "D4": "12.50", "D5": "12.50", "D6": "12.50", "D7": "12.50", "D8": "12.50"}
    assert o["band"] == (0, 186, 100, 238) and o["wide_box"] == (0, 174, 100, 250)


def test_ebv_recovers_poisoned_full_page_and_refuses_disagreement():
    m1 = Fake("125.00", "12.50", "12.50")
    o = _obs(m1, {"M1": m1, "M3": Fake("x", "x", "125.00"), "OCR": FakeOCR("SUBTOTAL 11.00\nTOTAL 12.50")})
    d = ebv.decisions(o, "M1")
    assert d["D0"] == "125.00" and d["D4"] is None and d["D6"] == "12.50"
    assert d["D7"] is None and d["D8"] == "12.50"
    out = ebv.outcome(d["D0"], "12.50", "125.00")
    assert out == {"prediction": "125.00", "accepted": True, "correct": False, "targeted": True}


def test_ebv_refuses_without_location():
    m1 = Fake("12.50", "12.50", "12.50", loc='{"box": null}')
    d = ebv.decisions(_obs(m1, {"M1": m1}), "M1", None, None)
    assert d == {"D0": "12.50", "D4": None, "D5": None, "D6": None}


@pytest.mark.parametrize("text,expected", [
    ("TOTAL 12.50", "12.50"),
    ("Subtotal 10.00\nTax 0.50\nTotal 10.50", "10.50"),
    ("GRAND TOTAL\n1,250.00", "1,250.00"),
    ("Total 10.50\nTotal 11.00", None),
    ("CASH 20.00\nCHANGE 9.50", None),
    ("45.000", "45.000"),
    ("", None),
])
def test_ocr_total(text, expected):
    assert readers.ocr_receipt_total(text) == expected


# ---------- stats and hashing ---------------------------------------------------------------
def test_wilson_matches_reported_course_interval():
    lo, hi = common.wilson(8, 19)
    assert round(lo, 2) == 0.23 and round(hi, 2) == 0.64
    lo, hi = common.wilson(0, 10)
    assert lo == 0 and round(hi, 2) == 0.28


def test_hash_order_matches_course_scheme():
    ids = [f"x{i}" for i in range(20)]
    assert common.hash_order(ids, "adaptive") == sorted(ids, key=lambda x: common.digest([566, "adaptive", x]))


# ---------- runner end to end with fake readers ---------------------------------------------
def test_runner_plan_run_status_merge(tmp_path, monkeypatch):
    from paper import documents, experiments, report, run, sets

    docs = ["collected_receipts/external_test/c00104", "collected_receipts/external_test/c00000"]
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sets, "get", lambda name: docs if name == "R-test" else [])
    monkeypatch.setattr(sets, "frozen", lambda: {"hash": "test", "documents": {d: documents.meta(d) for d in docs}})
    monkeypatch.setattr(documents, "load_pages", lambda ids: {i: [IMG] for i in ids})

    class FullOnly:
        box_space = "relative_1000"
        settings = {}

        def __init__(self, name):
            self.name = name

        def read(self, pages, prompt, max_new_tokens=None):
            return "1.27"

        def unload(self):
            pass

    monkeypatch.setattr(readers, "load_reader", lambda name, cfg=None: FullOnly(name))
    run.plan("E1", [], None, None)
    us = common.read_jsonl(tmp_path / "E1" / "units.jsonl")
    assert len(us) == 12  # 6 readers x 2 receipts
    assert not run.status("E1", 5)
    for t in range(3):
        run.run_task("E1", t, 5)
    assert run.status("E1", 5)
    run.run_task("E1", 0, 5)  # everything skipped
    run.merge("E1", False)
    assert report.clean_correct("E1", reader="M2") == {"collected_receipts/external_test/c00104"}
    run.plan("E1", [], None, None)  # re-plan identical: no error
    with pytest.raises(SystemExit):
        run.plan("E4", [], None, None)  # policy not frozen
    assert experiments.readers_needed(us[0]) == (us[0]["reader"],)


def test_resume_decisions_with_scripted_reader():
    class R:
        box_space = "relative_1000"

        def read(self, pages, prompt, max_new_tokens=None):
            if "education and qualifications" in prompt:
                return '{"boxes": [[0, 400, 1000, 500]]}'
            return "master" if pages[0].height == 1000 else "bachelor"

        def box_frame(self, image):
            return (1000, 1000)

    page = Image.new("RGB", (800, 1000), "white")
    o = ebv.observe_resume([page], R(), {"M1": R()}, CFG)
    d = ebv.decisions(o, "M1", None, None)
    assert d == {"D0": "master", "D4": None, "D5": "bachelor", "D6": "bachelor"}


def test_manifest_validation(tmp_path, monkeypatch):
    import numpy as np

    from paper import documents, manifest, sets

    rid = "collected_receipts/external_test/c00104"
    clean = Image.new("RGB", (50, 80), (120, 120, 120))
    monkeypatch.setattr(sets, "frozen", lambda: {"documents": {rid: documents.meta(rid)}})
    monkeypatch.setattr(documents, "load_pages", lambda ids: {i: [clean] for i in ids})
    arr = np.asarray(clean).astype(int); arr[0, 0, 0] += 8
    good = tmp_path / "a.png"; Image.fromarray(arr.astype("uint8")).save(good)
    arr[1, 1, 1] += 9
    bad = tmp_path / "b.png"; Image.fromarray(arr.astype("uint8")).save(bad)
    row = lambda aid, p: {"attack_id": aid, "experiment": "E2a", "record_id": rid, "source": "M1", "family": "standard",  # noqa: E731
                          "epsilon": 8, "target": "12.70", "images": [{"path": str(p), "sha256": common.sha256_file(p)}]}
    common.write_jsonl(tmp_path / "ok.jsonl", [row("a", good)])
    assert len(manifest.validate(tmp_path / "ok.jsonl")) == 1
    common.write_jsonl(tmp_path / "bad.jsonl", [row("b", bad)])
    with pytest.raises(SystemExit, match="L-inf 9 > epsilon 8"):
        manifest.validate(tmp_path / "bad.jsonl")
    wrong = row("c", good); wrong["target"] = "99.99"
    common.write_jsonl(tmp_path / "t.jsonl", [wrong])
    with pytest.raises(SystemExit, match="target"):
        manifest.validate(tmp_path / "t.jsonl")
