"""Frozen-reader input gradients for the paper attacks.

Qwen uses the course patch ordering. InternVL uses the installed Hugging Face
GotOCR tiling rule (grid fixed from clean dimensions), row-major tiles followed
by a thumbnail. Both use real quantized processor values forward and a smooth
resize/normalization backward (BPDA). Parity tests the *model logits*, including
the page-at-a-time vision path, against an independent official processor call.
No model weights are trained. GPU parity/memory measurements are an E0 gate,
not something the CPU tests can establish.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from PIL import Image

from attacks.donut import targeted_token_loss
from attacks.qwen import Objective as CourseQwenObjective, clear_rope
from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil

TOLERANCES = {"max": 1e-6, "mean": 1e-8, "surrogate_mean": 0.005}


class _OfficialForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate, official):
        if surrogate.shape != official.shape:
            raise ValueError("official and differentiable processor shapes differ")
        return official.to(surrogate).clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def official_forward(surrogate, official):
    # A custom function avoids cancellation/roundoff in x + (official - x).
    return _OfficialForward.apply(surrogate, official)


def enable_memory_saving(model):
    """Non-reentrant checkpointing supports frozen weights and input autograd.grad."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise TypeError("reader does not support gradient checkpointing")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


@contextmanager
def checkpoint_mode(model):
    """Activate HF checkpointing without enabling dropout or changing inference."""
    states = [(m, m.training) for m in model.modules()]
    # Only checkpoint-bearing modules need training=True. Dropout, batchnorm,
    # and the rest of the reader remain in deterministic evaluation mode.
    for m, _ in states:
        if getattr(m, "gradient_checkpointing", False):
            m.training = True
    try:
        yield
    finally:
        for m, state in states:
            m.training = state


def _join_features(values):
    """Support HF's tensor, per-image tuple, and ModelOutput feature APIs."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, (tuple, list)):
        return type(first)(x for value in values for x in value)
    if isinstance(first, dict):
        return type(first)(**{k: _join_features([v[k] for v in values]) for k in first})
    raise TypeError(f"unsupported image feature return type: {type(first)}")


@contextmanager
def pagewise_vision(model, counts, grids, grad_page):
    """Encode other pages under no_grad, retaining only the selected page graph.

The language model still sees *every* page in the original joint prompt. Calling
the whole loss once per page therefore gives the partials of the joint loss,
not the different objective obtained by reading each page independently.
"""
    if grad_page is None:
        yield
        return
    owners = [m for m in model.modules() if type(m).__name__ in ("Qwen2_5_VLModel", "InternVLModel")]
    if len(owners) != 1:
        raise RuntimeError("unsupported HF model layout for pagewise image features; run E0 in a compatible environment")
    owner = owners[0]
    original = owner.get_image_features
    previous = owner.__dict__.get("get_image_features")

    def split(_self, pixel_values, *args, **kwargs):
        if args:
            if grids is None or len(args) != 1:
                raise TypeError("unexpected image feature arguments")
            kwargs["image_grid_thw"] = args[0]
        results, offset = [], 0
        for i, count in enumerate(counts):
            kw = dict(kwargs)
            if grids is not None:
                kw["image_grid_thw"] = grids[i:i + 1]
            selected = pixel_values[offset:offset + count]
            offset += count
            with torch.set_grad_enabled(torch.is_grad_enabled() and i == grad_page):
                results.append(original(selected if i == grad_page else selected.detach(), **kw))
        if offset != pixel_values.shape[0]:
            raise ValueError("page/tile feature count changed")
        return _join_features(results)

    owner.get_image_features = MethodType(split, owner)
    try:
        yield
    finally:
        if previous is None:
            del owner.get_image_features
        else:
            owner.get_image_features = previous


class TargetObjective:
    """Shared teacher-forced target loss, parity and pagewise gradient API."""
    def _finish(self, target):
        tok = self.processor.tokenizer
        self.target_ids = tok.encode(target, add_special_tokens=False)
        if not self.target_ids:
            raise ValueError("empty target token sequence")
        eos = tok.eos_token_id
        if eos is None:
            raise ValueError("reader tokenizer has no EOS token")
        self.target_ids += [int(eos)]
        tail = torch.tensor([self.target_ids], device=self.device, dtype=torch.long)
        prefix = self.prefix["input_ids"]
        self.ids = torch.cat([prefix, tail], dim=1)
        self.positions = torch.arange(prefix.shape[1] - 1, self.ids.shape[1] - 1, device=self.device)
        self.target_tensor = tail.flatten()
        # Do not retain clean pixel tensors in every objective cache.
        self.prefix = {k: v for k, v in self.prefix.items() if k != "pixel_values"}

    def _batch(self, pixels):
        """Teacher-forced inputs. Per-token prefix tensors (Qwen's mm_token_type_ids in
        transformers 5.x) are extended over the appended target tokens as text (0)."""
        n = self.prefix["input_ids"].shape[1]
        extra = self.ids.shape[1] - n
        batch = {}
        for k, v in self.prefix.items():
            if k in ("input_ids", "attention_mask", "token_type_ids", "pixel_values"):
                continue
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == n:
                v = torch.cat([v, v.new_zeros((1, extra))], dim=1)
            batch[k] = v
        batch.update(input_ids=self.ids, attention_mask=torch.ones_like(self.ids),
                     pixel_values=pixels.to(self.model.dtype), use_cache=False)
        return batch

    def _logits(self, pixels, grad_page=None):
        clear_rope(self.model)
        batch = self._batch(pixels)
        with checkpoint_mode(self.model), pagewise_vision(self.model, self.counts, self.grids, grad_page):
            # Only answer-position logits are kept, avoiding a large retained
            # sequence-by-vocabulary tensor after this forward returns.
            return self.model(**batch, logits_to_keep=self.positions).logits

    def pixels(self, xs, grad_page=None):
        return official_forward(self.surrogate(xs, grad_page), self.official(xs))

    def loss(self, xs, grad_page=None):
        logits = self._logits(self.pixels(xs, grad_page), grad_page)
        positions = torch.arange(logits.shape[1], device=logits.device)
        return targeted_token_loss(logits, self.target_tensor, positions, loss_type="cross_entropy")

    @torch.no_grad()
    def parity(self, xs, tolerance=0.01):
        """Fail closed on processor drift, logit drift, or pagewise feature drift."""
        official = self.official(xs)
        approx = self.surrogate(xs)
        exact = self.pixels(xs)
        if not torch.isfinite(approx).all() or approx.shape != official.shape:
            raise RuntimeError("invalid differentiable processor output")
        processor_gap = float((exact - official.to(exact)).abs().max())
        approx_mean = float((approx - official.to(approx)).abs().mean())
        # Independent real processor reference, no feature splitting/checkpoint mode.
        clear_rope(self.model)
        batch = self._batch(official)
        reference = self.model(**batch, logits_to_keep=self.positions).logits.float().cpu()
        gap = float((self._logits(exact).float().cpu() - reference).abs().max())
        split_gap = float((self._logits(exact, 0).float().cpu() - reference).abs().max())
        stats = {"processor_max_abs_difference": processor_gap,
                 "processor_surrogate_mean_abs_difference": approx_mean,
                 "logit_max_abs_difference": gap, "pagewise_logit_max_abs_difference": split_gap,
                 "processor_shape": list(official.shape), "page_tile_counts": self.counts,
                 "gradient_method": "official_forward_bpda", "logit_tolerance": tolerance}
        stats["chat_processor_max_abs_difference"] = self.chat_processor_gap
        if (not all(torch.isfinite(torch.tensor(v)) for v in (gap, split_gap, approx_mean))
                or processor_gap > 1e-6 or approx_mean > self.surrogate_tolerance
                or max(gap, split_gap) > tolerance):
            raise RuntimeError(f"preprocessing/logit parity failed: {stats}")
        return stats


class QwenObjective(TargetObjective):
    def __init__(self, reader, pages, prompt, target):
        w = reader.w
        self.model, self.processor, self.device = w.model, w.processor, w.device
        self.core = CourseQwenObjective(w, pages, prompt, target, TOLERANCES)
        self.prefix, self.grids = dict(self.core.prefix), self.core.grids
        self.core.prefix = {}  # release the duplicate clean processor tensor
        self.counts = [int(t * h * w) for t, h, w in self.grids.tolist()]
        self.surrogate_tolerance = TOLERANCES["surrogate_mean"]
        check = self.official([pil_to_rgb_tensor(p, self.device) for p in pages])
        self.chat_processor_gap = float((check - self.prefix["pixel_values"].to(check)).abs().max())
        if self.chat_processor_gap > 1e-6:
            raise RuntimeError("Qwen chat/image processor mismatch")
        self._finish(target)

    def surrogate(self, xs, grad_page=None):
        inputs = [x if grad_page is None or i == grad_page else x.detach() for i, x in enumerate(xs)]
        return self.core._surrogate_pixels(inputs, grad_page)

    def official(self, xs):
        return self.core._official_pixels(xs)


class InternVLObjective(TargetObjective):
    def __init__(self, reader, pages, prompt, target):
        from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

        self.model, self.processor, self.device = reader.model, reader.processor, reader.device
        self.ip = self.processor.image_processor
        ip = self.ip
        if not (ip.do_resize and ip.do_rescale and ip.do_normalize):
            raise ValueError("unsupported InternVL processor flags")
        if abs(ip.rescale_factor - 1 / 255) > 1e-10 or int(ip.resample) != int(Image.Resampling.BICUBIC):
            raise ValueError("InternVL requires bicubic RGB rescale 1/255")
        self.size = (int(ip.size["height"]), int(ip.size["width"]))
        self.sizes = [p.size for p in pages]
        self.tiles = [get_optimal_tiled_canvas((p.height, p.width), self.size,
                                              ip.min_patches, ip.max_patches) for p in pages]
        self.counts = [w * h + int(w * h > 1) for w, h in self.tiles]
        content = [{"type": "image", "image": p} for p in pages] + [{"type": "text", "text": prompt}]
        self.prefix = dict(self.processor.apply_chat_template([{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device))
        if self.prefix["pixel_values"].shape[0] != sum(self.counts):
            raise RuntimeError("InternVL chat processor uses an unexpected tiling policy")
        self.grids = None
        self.surrogate_tolerance = 0.02  # PIL bicubic quantization, measured again in E0
        check = self.official([pil_to_rgb_tensor(p, self.device) for p in pages])
        self.chat_processor_gap = float((check - self.prefix["pixel_values"].to(check)).abs().max())
        if self.chat_processor_gap > 1e-6:
            raise RuntimeError("InternVL chat/image processor mismatch")
        self._finish(target)

    def surrogate(self, xs, grad_page=None):
        if len(xs) != len(self.tiles):
            raise ValueError("InternVL page count changed")
        h, w = self.size
        result = []
        for i, (x, (cols, rows)) in enumerate(zip(xs, self.tiles)):
            if (x.shape[-1], x.shape[-2]) != self.sizes[i]:
                raise ValueError("InternVL clean page dimensions changed")
            if grad_page is not None and i != grad_page:
                x = x.detach()
            canvas = F.interpolate(x[None], size=(rows * h, cols * w), mode="bicubic",
                                   align_corners=False, antialias=True).clamp(0, 1)[0]
            tiles = [canvas[:, r*h:(r+1)*h, c*w:(c+1)*w] for r in range(rows) for c in range(cols)]
            if rows * cols > 1:
                tiles.append(F.interpolate(x[None], size=(h, w), mode="bicubic",
                                           align_corners=False, antialias=True).clamp(0, 1)[0])
            y = torch.stack(tiles)
            mean = y.new_tensor(self.ip.image_mean).reshape(1, 3, 1, 1)
            std = y.new_tensor(self.ip.image_std).reshape(1, 3, 1, 1)
            result.append((y - mean) / std)
        return torch.cat(result)

    def official(self, xs):
        output = self.ip(images=[rgb_tensor_to_pil(x.detach()) for x in xs],
                         crop_to_patches=True, return_tensors="pt")
        counts = output.get("num_patches")
        if hasattr(counts, "tolist"):
            counts = counts.tolist()
        if list(counts) != self.counts:
            raise RuntimeError("InternVL tile counts drifted from the clean image")
        return output["pixel_values"].to(self.device)


def make_objective(reader, pages, prompt, target):
    if reader.name in ("M1", "M2"):
        return QwenObjective(reader, pages, prompt, target)
    if reader.name == "M3":
        return InternVLObjective(reader, pages, prompt, target)
    raise ValueError(f"no white-box surrogate for {reader.name}")


def page_gradients(objective, xs):
    """Exact joint-prompt partial derivatives, one page/backward at a time."""
    gradients, losses = [], []
    for i, x in enumerate(xs):
        loss, _ = objective.loss(xs, grad_page=i)
        gradient, = torch.autograd.grad(loss, x)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite target loss or image gradient")
        gradients.append(gradient.detach())
        losses.append(float(loss.detach()))
        del loss, gradient
    return sum(losses) / len(losses), gradients


class TrOCRObjective:
    """Differentiable OCR stand-in; success is always checked with real PaddleOCR."""
    def __init__(self, reader, pages, prompt, target):
        self.model, self.processor, self.device = reader.model, reader.processor, reader.device
        self.ip = self.processor.image_processor
        self.target_ids = self.processor.tokenizer.encode(target, add_special_tokens=True)
        self.labels = torch.tensor([self.target_ids], device=self.device)
        if len(pages) != 1 or not (self.ip.do_resize and self.ip.do_rescale and self.ip.do_normalize):
            raise ValueError("TrOCR surrogate expects one normalized band")

    def surrogate(self, xs):
        if len(xs) != 1:
            raise ValueError("TrOCR expects one band")
        ip, x = self.ip, xs[0]
        mode = {int(Image.Resampling.BILINEAR): "bilinear", int(Image.Resampling.BICUBIC): "bicubic"}.get(int(ip.resample))
        if mode is None or abs(ip.rescale_factor - 1 / 255) > 1e-10:
            raise ValueError("unsupported TrOCR resize/rescale")
        y = F.interpolate(x[None], size=(ip.size["height"], ip.size["width"]), mode=mode,
                          align_corners=False, antialias=True).clamp(0, 1)
        return (y - y.new_tensor(ip.image_mean)[None, :, None, None]) / y.new_tensor(ip.image_std)[None, :, None, None]

    def official(self, xs):
        return self.ip(images=[rgb_tensor_to_pil(x.detach()) for x in xs], return_tensors="pt")["pixel_values"].to(self.device)

    def _forward(self, pixels):
        with checkpoint_mode(self.model):
            return self.model(pixel_values=pixels.to(self.model.dtype), labels=self.labels, use_cache=False)

    def loss(self, xs, grad_page=None):
        output = self._forward(official_forward(self.surrogate(xs), self.official(xs)))
        return output.loss, -output.loss

    @torch.no_grad()
    def parity(self, xs, tolerance=0.01):
        official, surrogate = self.official(xs), self.surrogate(xs)
        reference = self._forward(official).logits.float().cpu()
        gap = float((self._forward(official_forward(surrogate, official)).logits.float().cpu() - reference).abs().max())
        approx = float((surrogate - official).abs().mean())
        if not torch.isfinite(torch.tensor([gap, approx])).all() or gap > tolerance or approx > 0.02:
            raise RuntimeError("TrOCR processor/logit parity failed")
        return {"logit_max_abs_difference": gap, "processor_surrogate_mean_abs_difference": approx,
                "gradient_method": "official_forward_bpda"}
