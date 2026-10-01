"""Run every vocab check over a vocab and return a report.

Each check is a property the gradient probe relies on, and each one guards
against a bug Sensaug actually had:

    shape_range       the output keeps the input's shape and stays finite and in [0, 1]
    identity          apply(x, identity_at) returns x
    moves_pixels      at the strongest magnitude in mag_range the image changes
    scalar_vs_vector  a scalar magnitude equals the same value given per image
    per_image         image i depends on its own magnitude alone: bumping image 0's
                      magnitude leaves the others bit-identical (for an op with no
                      magnitude: the batched call equals one image at a time)
    gradient          grad_mode "autograd" only: d(output)/d(magnitude) is (B,),
                      finite and nonzero for every image, and image 0's output has
                      exactly zero gradient w.r.t. the other images' magnitudes. An
                      op that collapses the per-image magnitude (e.g. delta.mean())
                      passes the shape check and fails this one.
    label             photometric: the label comes back unchanged. geometric: the
                      label moved with the image -- same shape, integer ids, no
                      invented class ids, and the class read off the warped image
                      agrees with the warped label (out of frame = ignore index)

Stochastic ops (`noise`) are run under a fixed seed, in a forked RNG so the audit
never consumes from the caller's streams, the same way the probe fixes its seed
per batch.

`audit` never raises for a bad op. A check that fails or errors is recorded as
"fail" with the reason, and a check that does not apply is recorded as "skip".
"""

import contextlib
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch

from .diff_ops import LABEL_IGNORE_INDEX
from .op import OpSpec

PASS, FAIL, SKIP = "pass", "fail", "skip"

CHECKS = (
    "shape_range",
    "identity",
    "moves_pixels",
    "scalar_vs_vector",
    "per_image",
    "gradient",
    "label",
)

# A quarter of one 8-bit level. kornia's identity warp carries float32 grid
# round-off that grows with image size (1e-5 at 48x64, 1.8e-4 at 1024x2048); a
# real zero-point bias, such as a half-pixel shift, is ~0.5 on a random texture.
IDENTITY_ATOL = 1e-3
UNCHANGED_ATOL = 1e-6  # "bit-identical" up to float noise
CHANGED_ATOL = 1e-4  # a change smaller than this does not count as moving
LABEL_AGREEMENT_MIN = 0.95  # bilinear image vs nearest label disagree on edges

# Distinct, far from black (the out-of-frame fill) and from each other, so every
# pixel's class can be read back off its colour after a warp.
_CLASS_COLOURS = torch.tensor(
    [[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]], dtype=torch.float32
) / 255.0


class _CheckFailed(Exception):
    pass


@dataclass(frozen=True)
class CheckResult:
    op: str
    check: str
    status: str
    detail: str = ""


@dataclass
class AuditReport:
    results: List[CheckResult]

    @property
    def ok(self) -> bool:
        return not self.failures()

    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    def status(self, op: str, check: str) -> str:
        for r in self.results:
            if r.op == op and r.check == check:
                return r.status
        raise KeyError((op, check))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "results": [asdict(r) for r in self.results]}

    def summary(self) -> str:
        ops = list(dict.fromkeys(r.op for r in self.results))
        cell = {(r.op, r.check): r.status for r in self.results}
        mark = {PASS: "ok", FAIL: "FAIL", SKIP: "-"}
        name_w = max([len("op")] + [len(o) for o in ops])
        widths = [max(len(c), 4) for c in CHECKS]
        lines = ["  ".join(["op".ljust(name_w)] + [c.ljust(w) for c, w in zip(CHECKS, widths)])]
        for op in ops:
            row = [op.ljust(name_w)]
            row += [mark[cell[(op, c)]].ljust(w) for c, w in zip(CHECKS, widths)]
            lines.append("  ".join(row).rstrip())
        failures = self.failures()
        lines.append(f"{len(ops)} ops, {len(failures)} failed checks")
        lines += [f"  {r.op}.{r.check}: {r.detail}" for r in failures]
        return "\n".join(lines)


# --- helpers -----------------------------------------------------------------


@contextlib.contextmanager
def _seeded(seed: int, device: torch.device):
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def _strongest(spec: OpSpec) -> float:
    """The end of mag_range farthest from the identity, toward `hi` on a tie.

    The same rule as the probe's `strength_to_magnitude(op, 1.0)` (the probe does
    not import vocab, so it keeps its own copy; tests/probe/test_vocab_ops.py pins
    the two together). A tie up to float error is a tie: for the enhance ops,
    1.0 - 0.1 exceeds 1.9 - 1.0 by one ulp."""
    lo, hi = spec.mag_range
    anchor = spec.identity_at
    below_is_farther = (anchor - lo) - (hi - anchor) > 1e-9 * (hi - lo)
    return lo if below_is_farther else hi


def _midway(spec: OpSpec) -> float:
    return spec.identity_at + 0.5 * (_strongest(spec) - spec.identity_at)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def _expect(condition: bool, detail: str) -> None:
    if not condition:
        raise _CheckFailed(detail)


def _class_scene(batch_size: int, size: Tuple[int, int], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """An image whose pixel classes can be read back off its colours: four
    quadrants, one class each, and the matching (B, H, W) label."""
    h, w = size
    label = torch.zeros(h, w, dtype=torch.long)
    label[: h // 2, w // 2 :] = 1
    label[h // 2 :, : w // 2] = 2
    label[h // 2 :, w // 2 :] = 3
    image = _CLASS_COLOURS[label].permute(2, 0, 1)  # (3, H, W)
    images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    labels = label.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    return images, labels


def _read_classes(images: torch.Tensor) -> torch.Tensor:
    """Nearest class colour per pixel, with black (the out-of-frame fill) read as
    the ignore index."""
    palette = torch.cat([_CLASS_COLOURS, torch.zeros(1, 3)]).to(images.device)  # (K+1, 3)
    pixels = images.detach().permute(0, 2, 3, 1).unsqueeze(-2)  # (B, H, W, 1, 3)
    nearest = (pixels - palette).pow(2).sum(-1).argmin(-1)  # (B, H, W)
    return torch.where(nearest == len(_CLASS_COLOURS), torch.full_like(nearest, LABEL_IGNORE_INDEX), nearest)


# --- checks ------------------------------------------------------------------


class _Auditor:
    def __init__(self, batch_size: int, image_size: Tuple[int, int], seed: int, device):
        self.device = torch.device(device)
        self.seed = seed
        self.batch_size = batch_size
        generator = torch.Generator().manual_seed(seed)
        # 8-bit-representable, so ops that run on 8-bit images are exact on it.
        self.images = (
            torch.randint(0, 256, (batch_size, 3, *image_size), generator=generator).float() / 255.0
        ).to(self.device)
        self.scene, self.scene_label = _class_scene(batch_size, image_size, self.device)

    def _apply(self, spec, images, magnitude, label=None):
        with _seeded(self.seed, self.device):
            return spec.apply(images, magnitude, label)

    def _vector(self, value: float, requires_grad: bool = False) -> torch.Tensor:
        return torch.full(
            (self.batch_size,), float(value), device=self.device, requires_grad=requires_grad
        )

    def shape_range(self, spec: OpSpec) -> str:
        magnitude = _midway(spec) if spec.has_magnitude else None
        out, _ = self._apply(spec, self.images, magnitude)
        _expect(out.shape == self.images.shape, f"shape {tuple(out.shape)} != {tuple(self.images.shape)}")
        _expect(bool(torch.isfinite(out).all()), "non-finite output")
        lo, hi = float(out.min()), float(out.max())
        _expect(lo >= 0.0 and hi <= 1.0, f"output range [{lo:.4g}, {hi:.4g}] leaves [0, 1]")
        return ""

    def identity(self, spec: OpSpec) -> str:
        if not spec.has_magnitude:
            return SKIP, "no magnitude"
        out, _ = self._apply(spec, self.images, spec.identity_at)
        diff = _max_diff(out, self.images)
        _expect(diff <= IDENTITY_ATOL, f"max |out - in| = {diff:.3g} at identity_at={spec.identity_at}")
        return f"max diff {diff:.2g}"

    def moves_pixels(self, spec: OpSpec) -> str:
        if not spec.has_magnitude:
            return SKIP, "no magnitude"
        strongest = _strongest(spec)
        out, _ = self._apply(spec, self.images, strongest)
        diff = _max_diff(out, self.images)
        _expect(diff > CHANGED_ATOL, f"image unchanged at magnitude {strongest}")
        return f"at {strongest}"

    def scalar_vs_vector(self, spec: OpSpec) -> str:
        if not spec.has_magnitude:
            return SKIP, "no magnitude"
        m = _midway(spec)
        scalar, _ = self._apply(spec, self.images, m)
        vector, _ = self._apply(spec, self.images, self._vector(m))
        diff = _max_diff(scalar, vector)
        _expect(diff <= UNCHANGED_ATOL, f"scalar and per-image magnitude differ by {diff:.3g}")
        return ""

    def per_image(self, spec: OpSpec) -> str:
        if not spec.has_magnitude:
            batched, _ = self._apply(spec, self.images, None)
            for i in range(self.batch_size):
                alone, _ = self._apply(spec, self.images[i : i + 1], None)
                diff = _max_diff(batched[i : i + 1], alone)
                _expect(diff <= UNCHANGED_ATOL, f"image {i} differs by {diff:.3g} batched vs alone")
            return "batched == one at a time"
        base = self._vector(_midway(spec))
        bumped = base.clone()
        bumped[0] = _strongest(spec)
        out_base, _ = self._apply(spec, self.images, base)
        out_bumped, _ = self._apply(spec, self.images, bumped)
        others = _max_diff(out_base[1:], out_bumped[1:])
        _expect(others <= UNCHANGED_ATOL, f"bumping image 0's magnitude moved the others by {others:.3g}")
        own = _max_diff(out_base[:1], out_bumped[:1])
        _expect(own > CHANGED_ATOL, "bumping image 0's magnitude did not change image 0")
        return ""

    def gradient(self, spec: OpSpec) -> str:
        if spec.grad_mode != "autograd":
            return SKIP, f"grad_mode={spec.grad_mode!r}"
        # Geometric ops go through the label path, which is what the probe calls.
        images = self.scene if spec.warps_label else self.images
        label = self.scene_label if spec.warps_label else None
        weights = torch.arange(1, self.batch_size + 1, dtype=images.dtype, device=self.device)

        delta = self._vector(_midway(spec), requires_grad=True)
        out, _ = self._apply(spec, images, delta, label)
        loss = (out * weights.view(-1, 1, 1, 1)).sum()
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        _expect(grad is not None, "magnitude is not part of the graph")
        _expect(tuple(grad.shape) == (self.batch_size,), f"gradient shape {tuple(grad.shape)}")
        _expect(bool(torch.isfinite(grad).all()), f"non-finite gradient {grad.tolist()}")
        _expect(bool((grad != 0).all()), f"zero gradient for some image: {grad.tolist()}")

        delta = self._vector(_midway(spec), requires_grad=True)
        out, _ = self._apply(spec, images, delta, label)
        cross = torch.autograd.grad(out[0].sum(), delta)[0]
        _expect(float(cross[0]) != 0.0, "image 0 has no gradient w.r.t. its own magnitude")
        _expect(
            bool((cross[1:] == 0).all()),
            f"image 0's output has gradient w.r.t. other images' magnitudes: {cross.tolist()}",
        )
        return ""

    def label(self, spec: OpSpec) -> str:
        magnitude = _strongest(spec) if spec.has_magnitude else None
        out, label_out = self._apply(spec, self.scene, magnitude, self.scene_label)
        if not spec.warps_label:
            _expect(
                label_out is not None and torch.equal(label_out, self.scene_label),
                "photometric op changed the label",
            )
            return "unchanged"
        _expect(label_out is not None, "geometric op returned no label")
        _expect(
            tuple(label_out.shape) == tuple(self.scene_label.shape),
            f"label shape {tuple(label_out.shape)} != {tuple(self.scene_label.shape)}",
        )
        _expect(not label_out.is_floating_point(), f"label dtype {label_out.dtype} is not integral")
        _expect(not torch.equal(label_out, self.scene_label), "label did not move with the image")
        invented = set(label_out.unique().tolist()) - set(self.scene_label.unique().tolist()) - {
            LABEL_IGNORE_INDEX
        }
        _expect(not invented, f"label contains invented class ids {sorted(invented)}")
        agreement = float((_read_classes(out) == label_out).float().mean())
        _expect(
            agreement >= LABEL_AGREEMENT_MIN,
            f"image and label agree on only {agreement:.1%} of pixels",
        )
        return f"agreement {agreement:.1%}"

    def run(self, spec: OpSpec) -> List[CheckResult]:
        results = []
        for check in CHECKS:
            try:
                outcome = getattr(self, check)(spec)
                status, detail = outcome if isinstance(outcome, tuple) else (PASS, outcome)
            except _CheckFailed as e:
                status, detail = FAIL, str(e)
            except Exception as e:  # a crashing op is a failed check, not a crashed audit
                status, detail = FAIL, f"{type(e).__name__}: {e}"
            results.append(CheckResult(spec.name, check, status, detail))
        return results


def audit(
    vocab: Optional[Union[Mapping[str, OpSpec], Iterable[OpSpec]]] = None,
    *,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (48, 64),
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> AuditReport:
    """Run every check in CHECKS on every op of `vocab` (a `{name: OpSpec}`
    mapping or an iterable of OpSpecs; default: every op in every family)."""
    if vocab is None:
        from .registry import OPS

        vocab = OPS
    specs = list(vocab.values()) if isinstance(vocab, Mapping) else list(vocab)
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 to check per-image independence")
    auditor = _Auditor(batch_size, image_size, seed, device)
    results: List[CheckResult] = []
    for spec in specs:
        results.extend(auditor.run(spec))
    return AuditReport(results)
