"""CPU contracts, real processor/tiny-model gradients and crash recovery.

No pretrained weights or network access; E0 must still validate pinned GPU models.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attacks import paper_protocol as protocol
from attacks.paper_optimizer import optimize
from attacks.surrogates import (InternVLObjective, QwenObjective, enable_memory_saving,
                                official_forward, page_gradients)
from models.donut import pil_to_rgb_tensor


def dataset(name):
    n = {"R-attack": 100, "Res-attack": 50, "R-direct": 50,
         "Res-direct": 30, "R-conv": 20}[name]
    return [("res" if name.startswith("Res") else "r") + str(i) for i in range(n)]


def metadata(rid):
    resume = rid.startswith("res")
    return {"task": "resume_highest_degree" if resume else "receipt_total",
            "dataset_name": "resume_parsing_vision" if resume else "sroie",
            "clean_target": "bachelor" if resume else "12.50",
            "attack_target": "postgraduate" if resume else "125.00"}


@pytest.mark.parametrize("experiment,count,rows", [
    ("E2a", 1200, 1200), ("E2b", 400, 400), ("E2c", 300, 300),
    ("E3b", 100, 100), ("E5a", 60, 180), ("E5b", 450, 450),
    ("E5c", 60, 60), ("E6", 120, 120), ("E7", 250, 250),
])
def test_frozen_schedule(experiment, count, rows):
    units = protocol.build_units(experiment, {}, protocol.settings(), dataset, metadata, 250)
    assert len(units) == count
    assert sum(len(u.get("milestones", [u["steps"]])) for u in units) == rows
    assert len({u["attack_id"] for u in units}) == count
    assert units == protocol.build_units(experiment, {}, protocol.settings(), dataset, metadata, 250)


def test_e7_targets_are_wrong_and_normalized():
    from normalization import normalize_receipt_total, normalize_resume_degree

    for ds, clean, target in [("cord_v2", "2000", "20000"), ("sroie", "0.00", "0.00"),
                               ("collected_receipts", "731.00", "7310.00")]:
        variants = dict(protocol.receipt_targets({"dataset_name": ds, "clean_target": clean, "attack_target": target}))
        assert variants["digit_flip"] != clean
        assert variants["unrelated"] not in (clean, target, variants["digit_flip"])
        assert all(normalize_receipt_total(t, ds) == t for t in variants.values())
    units = protocol.build_units("E7", {}, protocol.settings(), dataset, metadata)
    for u in units:
        if u["task"] == "resume":
            assert normalize_resume_degree(u["target"]) == u["target"] != metadata(u["record_id"])["clean_target"]


def test_budget_counts_documents_not_restarts():
    rows = [{"unit": {"record_id": rid}, "snapshots": [
        {"steps": n, "verification": {"success": n >= threshold}} for n in (100, 250, 500)]}
        for rid, threshold in [("a", 100), ("a", 100), ("b", 250), ("c", 500)]]
    assert protocol.choose_budget(rows)["steps"] == 250
    assert protocol.choose_budget(rows)["documents_broken"] == {"100": 1, "250": 2, "500": 3}


def test_bpda_exact_values_and_gradient():
    x = torch.randn(5, requires_grad=True)
    official = torch.randn(5)
    y = official_forward(x, official)
    assert torch.equal(y, official)
    assert torch.equal(torch.autograd.grad(y.sum(), x)[0], torch.ones_like(x))
    with pytest.raises(ValueError):
        official_forward(x, torch.ones(3))


class Tokenizer:
    eos_token_id = 5

    def encode(self, target, **kw):
        return [4]  # one teacher-forced target token plus EOS


def tiny_reader(name):
    """Real HF architectures/processors with tiny random weights and a fake chat prefix."""
    transformers = pytest.importorskip("transformers")
    from transformers.image_processing_utils import BatchFeature
    torch.manual_seed(123)
    text = {"model_type": "qwen2" if name == "M1" else "llama", "vocab_size": 64,
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 1024}
    if name == "M1":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        text["rope_scaling"] = {"type": "mrope", "mrope_section": [1, 1, 2]}
        cfg = transformers.Qwen2_5_VLConfig(text_config=text,
            vision_config={"depth": 1, "hidden_size": 16, "num_heads": 2, "intermediate_size": 32,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "window_size": 8, "fullatt_block_indexes": [0]},
            image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13)
        model = transformers.Qwen2_5_VLForConditionalGeneration(cfg).eval()
        ip = Qwen2VLImageProcessor(min_pixels=64, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2)
    else:
        from transformers.models.got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor
        cfg = transformers.InternVLConfig(text_config=text, image_token_id=10, image_seq_length=4,
            downsample_ratio=0.5, vision_config={"hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": [8, 8], "patch_size": [2, 2]})
        model = transformers.InternVLForConditionalGeneration(cfg).eval()
        ip = GotOcr2ImageProcessor(size={"height": 8, "width": 8}, min_patches=1, max_patches=2)

    class Processor:
        image_processor, tokenizer = ip, Tokenizer()

        def apply_chat_template(self, messages, **kw):
            if name == "M1":
                return "chat"
            pages = [c["image"] for c in messages[0]["content"] if c["type"] == "image"]
            return self(images=pages)

        def __call__(self, images, **kw):
            features = ip(images=images, return_tensors="pt", **({"crop_to_patches": True} if name == "M3" else {}))
            if name == "M1":
                counts = [int(t*h*w)//4 for t, h, w in features["image_grid_thw"].tolist()]
            else:
                counts = [int(n)*4 for n in features.pop("num_patches")]
            ids = [1]
            for n in counts:
                ids += [12] + [10] * n + [13]
            ids += [2]
            features["input_ids"] = torch.tensor([ids])
            features["attention_mask"] = torch.ones_like(features["input_ids"])
            return BatchFeature(features)

    enable_memory_saving(model)
    wrapper = SimpleNamespace(model=model, processor=Processor(), device=torch.device("cpu"))
    return SimpleNamespace(name=name, w=wrapper) if name == "M1" else SimpleNamespace(name=name, **vars(wrapper))


@pytest.mark.parametrize("name", ["M1", "M3"])
def test_real_tiny_model_pagewise_parity_and_gradients(name):
    torch.set_num_threads(1)
    reader = tiny_reader(name)
    rng = np.random.default_rng(8)
    pages = [Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)),
             Image.fromarray(rng.integers(0, 256, (16, 8, 3), dtype=np.uint8))]
    obj = (QwenObjective if name == "M1" else InternVLObjective)(reader, pages, "read", "4")
    xs = [pil_to_rgb_tensor(p, torch.device("cpu")).requires_grad_(True) for p in pages]
    stats = obj.parity(xs)
    assert stats["logit_max_abs_difference"] <= .01
    loss, _ = obj.loss(xs)
    joint = torch.autograd.grad(loss, xs)
    page_loss, separate = page_gradients(obj, xs)
    assert page_loss == pytest.approx(float(loss.detach()), abs=1e-5)
    for a, b in zip(joint, separate):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
        assert float(b.abs().sum()) > 0
    if name == "M3":
        assert obj.counts == [1, 3]
        assert len(obj.tiles) == 2


class ToyObjective:
    def __init__(self, crash_at=None):
        self.crash_at = crash_at
        self.cache = SimpleNamespace(diagnostics=[{"test": True}])

    def gradient(self, xs, step, *args):
        if step == self.crash_at:
            raise RuntimeError("simulated interruption")
        return sum(float(x.mean()) for x in xs), [torch.ones_like(x) for x in xs]

    def score(self, xs):
        return sum(float(x.mean()) for x in xs), None

    def verify(self, pages):
        # Always unsuccessful to exercise the complete real-query budget.
        return {"success": False, "target_count": 0, "prediction": "clean"}


def toy_unit(family="standard", milestones=False):
    return {"unit_id": "12345678abcdef0000000000", "attack_id": "test", "family": family,
            "steps": 6, "epsilon": 8, "experiment": "E5a" if milestones else "E2a",
            "record_id": "test", "source": "M1", "target": "99", "restart": 0,
            **({"milestones": [2, 4, 6]} if milestones else {})}


def toy_settings():
    return {"refresh_interval": 2, "evaluation_interval": 2, "checkpoint_interval": 2,
            "queries": 12, "step_size_255": {"8": 2.0}}


@pytest.mark.parametrize("family", ["standard", "direct"])
def test_checkpoint_resume_matches_uninterrupted_pngs(tmp_path, family):
    pages = [Image.new("RGB", (9, 7), "gray"), Image.new("RGB", (5, 11), "gray")]
    unit, settings = toy_unit(family, family == "direct"), toy_settings()
    continuous = optimize(unit, pages, ToyObjective(), settings, tmp_path / "one", "contract", "cpu")
    with pytest.raises(RuntimeError, match="simulated"):
        optimize(unit, pages, ToyObjective(3), settings, tmp_path / "two", "contract", "cpu")
    resumed = optimize(unit, pages, ToyObjective(), settings, tmp_path / "two", "contract", "cpu")
    assert resumed["trace"] == continuous["trace"]
    assert resumed["queries"] == continuous["queries"]
    assert [[i["sha256"] for i in s["images"]] for s in resumed["snapshots"]] == [
        [i["sha256"] for i in s["images"]] for s in continuous["snapshots"]]
    assert all(i["linf_255"] <= 8.0001 for s in resumed["snapshots"] for i in s["images"])
    if family == "direct":
        assert resumed["queries"] == settings["queries"]
        assert [s["steps"] for s in resumed["snapshots"]] == [2, 4, 6]
    with pytest.raises(RuntimeError, match="contract changed"):
        optimize(unit, pages, ToyObjective(), settings, tmp_path / "two", "different", "cpu")
    with pytest.raises(RuntimeError, match="contract changed"):
        optimize(unit, [Image.new("RGB", (9, 7), "white"), pages[1]], ToyObjective(), settings,
                 tmp_path / "two", "contract", "cpu")


def test_saved_manifest_passes_real_pixel_validator(tmp_path, monkeypatch):
    from paper import manifest, sets, documents

    pages = [Image.new("RGB", (8, 8), "gray")]
    result = optimize(toy_unit(), pages, ToyObjective(), toy_settings(), tmp_path, "c", "cpu")
    monkeypatch.setattr(sets, "frozen", lambda: {"documents": {"test": {"attack_target": "99"}}})
    monkeypatch.setattr(documents, "load_pages", lambda ids: {"test": pages})
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(result["snapshots"][0]) + "\n")
    assert len(manifest.validate(path)) == 1


def test_query_budget_rejected_before_running(tmp_path):
    with pytest.raises(ValueError, match="query budget"):
        optimize(toy_unit("direct", True), [Image.new("RGB", (8, 8))], ToyObjective(),
                 {**toy_settings(), "queries": 2}, tmp_path, "c", "cpu")
