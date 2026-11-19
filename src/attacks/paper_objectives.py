"""Model residency, saved/EOT objectives and adaptive EBV/locator objectives."""
from __future__ import annotations

from collections import OrderedDict
import gc
import json

import numpy as np
import torch

from attacks.jpeg_eot import (_transformed_pages_bpda, hard_view_weights,
                              resolved_training_views, view_schedule)
from attacks.surrogates import (TrOCRObjective, enable_memory_saving,
                                make_objective, page_gradients)
from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil
from paper import ebv
from paper.common import digest
from paper.experiments import _normalizer
from paper.readers import load_reader, ocr_receipt_total


class ReaderPool:
    """Keep only one gradient model on the GPU; sum losses/gradients sequentially."""
    def __init__(self, cfg, settings):
        self.cfg, self.settings, self.readers = cfg, settings, {}
        self.active = None

    def get(self, name):
        if name != self.active and name != "OCR":
            if self.active in self.readers:
                old = self.readers[self.active]
                (old.w.model if hasattr(old, "w") else old.model).to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.active = name
        if name not in self.readers:
            if name == "S":
                from types import SimpleNamespace
                from transformers import AutoProcessor, VisionEncoderDecoderModel

                spec = self.settings["ocr_surrogate"]
                if spec["revision"] == "PIN_IN_E0":
                    raise RuntimeError("pin the TrOCR revision in the attack settings before D8")
                device = torch.device("cuda")
                model = VisionEncoderDecoderModel.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                                  torch_dtype=torch.bfloat16).to(device).eval()
                reader = SimpleNamespace(name="S", model=model, device=device,
                    processor=AutoProcessor.from_pretrained(spec["model_id"], revision=spec["revision"]), settings=spec)
            else:
                reader = load_reader(name, self.cfg)
            if name != "OCR":
                enable_memory_saving(reader.w.model if hasattr(reader, "w") else reader.model)
            self.readers[name] = reader
        reader = self.readers[name]
        if name != "OCR":
            (reader.w.model if hasattr(reader, "w") else reader.model).to("cuda")
        return reader

    def proxy(self, name):
        pool = self

        class Proxy:
            def read(self, *args, **kwargs):
                return pool.get(name).read(*args, **kwargs)

            @property
            def box_space(self):
                return pool.get(name).box_space

            def box_frame(self, image):
                return pool.get(name).box_frame(image)

        return Proxy()

    def close(self):
        for reader in self.readers.values():
            if hasattr(reader, "unload"):
                reader.unload()
            else:
                del reader.model, reader.processor
        self.readers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ObjectiveCache:
    def __init__(self, pool, settings):
        self.pool, self.settings = pool, settings
        self.cache, self.diagnostics = OrderedDict(), []

    def get(self, name, xs, prompt, target):
        reader = self.pool.get(name)
        key = (name, tuple(tuple(x.shape) for x in xs), prompt, target)
        if key not in self.cache:
            pages = [rgb_tensor_to_pil(x.detach()) for x in xs]
            ctor = TrOCRObjective if name == "S" else make_objective
            obj = ctor(reader, pages, prompt, target)
            stats = obj.parity(xs, tolerance=self.settings["logit_tolerance"])
            self.diagnostics.append({"source": name, "shapes": [list(x.shape) for x in xs],
                "target": target, "target_token_ids": obj.target_ids,
                "target_tokens": len(obj.target_ids), **stats})
            if len(self.cache) >= 8:
                self.cache.popitem(last=False)
            self.cache[key] = obj
        self.cache.move_to_end(key)
        return self.cache[key]


class SavedObjective:
    def __init__(self, unit, cfg, settings, pool):
        self.u, self.cfg, self.settings, self.pool = unit, cfg, settings, pool
        self.cache = ObjectiveCache(pool, settings)
        self.names = unit["source"].split("+")
        self.prompt = cfg["prompts"][unit["task"]]
        self.norm = _normalizer(unit["record_id"])
        self.views = [{"transform": "identity", "prompt_index": 0}]
        self.prompts = [self.prompt]
        self.eot = settings.get(unit["family"]) if unit["family"] in ("uniform_eot", "hardview") else None
        if self.eot:
            self.views = resolved_training_views(self.eot, unit["record_id"])
            self.prompts = list(self.eot["semantic_prompts"][unit["task"]])
            self.prompts[0] = self.prompt

    def gradient(self, xs, step, observation=None, weights=None):
        selected = (0,)
        if self.eot:
            selected = view_schedule(len(self.views), self.eot["attack"]["eot_samples_per_step"],
                seed=int(self.u["unit_id"][:8], 16), step=step,
                weights=weights if self.u["family"] == "hardview" else None)
        total, gradients = 0., [torch.zeros_like(x) for x in xs]
        for name in self.names:
            for index in selected:
                views = _transformed_pages_bpda(xs, self.views[index])
                obj = self.cache.get(name, views, self.prompts[self.views[index]["prompt_index"]], self.u["target"])
                loss, gs = page_gradients(obj, views)
                total += loss
                # All course EOT transformations use identity backward BPDA.
                for g, part in zip(gradients, gs):
                    g.add_(part)
        count = len(self.names) * len(selected)
        return total / count, [g / count for g in gradients]

    @torch.no_grad()
    def score(self, xs):
        losses = []
        for index, view in enumerate(self.views):
            per_model = []
            for name in self.names:
                transformed = _transformed_pages_bpda(xs, view)
                obj = self.cache.get(name, transformed, self.prompts[view["prompt_index"]], self.u["target"])
                per_model.append(float(obj.loss(transformed)[0]))
            losses.append(sum(per_model) / len(per_model))
        weights = None
        if self.u["family"] == "hardview":
            s = self.eot["attack"]
            weights = hard_view_weights(losses, temperature=s["hard_view_temperature"], floor=s["hard_view_probability_floor"])
        return sum(losses) / len(losses), weights

    def verify(self, pages):
        predictions = {name: self.norm(self.pool.get(name).read(pages, self.prompt)) for name in self.names}
        if self.eot:
            from defenses import evaluate_defense, evaluate_semantic_consistency

            name = self.names[0]
            for defense in ("jpeg", "resize_vote", "selective_multiview"):
                result, _ = evaluate_defense(pages=pages, defense_id=defense,
                    record_id=self.u["record_id"], predict=lambda p: self.pool.get(name).read(p, self.prompt),
                    normalize=self.norm, config=self.eot["defense"])
                predictions[defense] = None if result.abstained else result.final_prediction
            result, _ = evaluate_semantic_consistency(pages=pages, record_id=self.u["record_id"],
                predict=lambda p, q: self.pool.get(name).read(p, q),
                normalize=self.norm, prompts=self.prompts,
                transforms=self.eot["defense"]["semantic_transforms"],
                minimum_votes=self.eot["defense"]["semantic_minimum_votes"])
            predictions["semantic"] = None if result.abstained else result.final_prediction
        count = sum(p == self.u["target"] for p in predictions.values())
        return {"predictions": predictions, "success": count == len(predictions), "target_count": count}


class DirectObjective:
    def __init__(self, unit, cfg, settings, pool, clean_pages, locator_reference=None):
        self.u, self.cfg, self.settings, self.pool = unit, cfg, settings, pool
        self.cache = ObjectiveCache(pool, settings)
        self.prompt = cfg["prompts"][unit["task"]]
        self.method = unit["attacked_method"]
        self.locator_reference = locator_reference
        if self.method == "locator" and self.locator_reference is None:
            self.locator_reference = self.prepare_locator(clean_pages)

    def observe(self, pages):
        primary = self.pool.proxy("M1")
        wide = {"D7": "M3", "D8": "OCR"}.get(self.method, "M1")
        readers = {wide: self.pool.proxy(wide)}
        if self.u["task"] == "resume":
            return ebv.observe_resume(pages, primary, readers, self.cfg)
        from paper import documents

        return ebv.observe_receipt(pages[0], documents.meta(self.u["record_id"])["dataset_name"],
                                   primary, readers, self.cfg, ocr_receipt_total)

    def verify(self, pages):
        obs = self.observe(pages)
        method = "D6" if self.method == "locator" else self.method
        prediction = ebv.decisions(obs, "M1").get(method)
        success = prediction == self.u["target"]
        subtotal_overlap = None
        if self.method == "locator":
            if self.u["locator_goal"] == "refusal":
                success = prediction is None
            else:
                box, reference = obs.get("tight_box"), self.locator_reference["subtotal_box"]
                subtotal_overlap = box_iou(box, reference)
                # Subtotal steering is diagnostic; a forgery still requires
                # the exact frozen 10x answer accepted by the actual D6 pipeline.
                success = success and subtotal_overlap >= 0.5
        count = int(obs["crop"] == self.u["target"]) + sum(v == self.u["target"] for v in obs["wide"].values())
        return {"prediction": prediction, "success": success, "target_count": count,
                "refused": prediction is None, "subtotal_iou": subtotal_overlap, "observation": obs}

    def prepare_locator(self, pages):
        reader = self.pool.get("M1")
        if self.u["locator_goal"] == "refusal":
            # An off-page box is invalid under the real parser; optimizing
            # its tokens targets a concrete refusal rather than answer loss.
            frame = reader.box_frame(pages[0])
            return {"target_text": json.dumps({"box": [0, frame[1] + 1, frame[0], frame[1] + 30]}),
                    "subtotal_box": None}
        raw = reader.read(pages, self.settings["locator_subtotal_prompt"], 96)
        frame = reader.box_frame(pages[0])
        box = ebv.receipt_parse_box(raw, frame, pages[0].size, self.cfg["ebv_policy"]["receipt"])
        if box is None:
            raise ValueError("clean subtotal localization unavailable; record failure, do not invent a box")
        # Freeze a canonical clean reference; use no attacked prediction as a label.
        w, h = pages[0].size
        coords = [round(v * d / s) for v, d, s in zip(box, (*frame, *frame), (w, h, w, h))]
        return {"target_text": json.dumps({"box": coords}), "subtotal_box": list(box), "clean_subtotal_raw": raw}

    def groups(self, observation, page_count):
        full = [(j, None) for j in range(page_count)]
        if self.method == "locator" and self.u["locator_goal"] == "refusal":
            return []
        if self.u["task"] == "resume":
            regions = (observation or {}).get("regions", [])
            tight = [(r["page"], b) for r in regions for b, e in r["pairs"]]
            wide = [(r["page"], e) for r in regions for b, e in r["pairs"]]
        else:
            tight_box = (observation or {}).get("band")
            wide_box = (observation or {}).get("wide_box")
            if self.method == "locator" and self.u["locator_goal"] == "subtotal":
                tight_box = ebv.receipt_band(self.locator_reference["subtotal_box"],
                    self.page_size, self.cfg["ebv_policy"]["receipt"])
                wide_box = ebv.receipt_expand(tight_box, self.page_size, self.cfg["ebv_policy"]["receipt"])
            tight = [(0, tight_box)] if tight_box else []
            wide = [(0, wide_box)] if wide_box else []
        if not tight or not wide:
            return [("M1", full)]
        if self.method == "D4":
            return [("M1", full), ("M1", tight)]
        second = {"D7": "M3", "D8": "S"}.get(self.method, "M1")
        return [("M1", tight), (second, wide)]

    def gradient(self, xs, step, observation=None, weights=None):
        self.page_size = (xs[0].shape[-1], xs[0].shape[-2])
        total, grads = 0., [torch.zeros_like(x) for x in xs]
        terms = [(name, group, self.prompt, self.u["target"])
                 for name, group in self.groups(observation, len(xs))]
        if self.method == "locator":
            terms.append(("M1", [(0, None)], self.cfg["prompts"]["receipt_localize_relative"],
                          self.locator_reference["target_text"]))
        for name, group, prompt, target in terms:
            views = [xs[j] if b is None else xs[j][:, b[1]:b[3], b[0]:b[2]] for j, b in group]
            obj = self.cache.get(name, views, prompt, target)
            loss, gs = page_gradients(obj, views)
            total += loss
            for (j, box), g in zip(group, gs):
                if box is None:
                    grads[j].add_(g)
                else:
                    grads[j][:, box[1]:box[3], box[0]:box[2]].add_(g)
        return total / len(terms), [g / len(terms) for g in grads]


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area - intersection) if area > intersection else 0.
