"""Ternary expert blocks for AetherCore v3.

The expert block is a lightweight PyTorch module built around ternary
``{-1, 0, +1}`` weights, a scale factor, and an optional tiny correction table.
It is designed to be serializable, CPU-friendly, and compatible with the
correction-table shape produced by ``src.compression.engine``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from src.compression.engine import CorrectionTableExtractor, TernaryQuantizer
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.compression.engine import CorrectionTableExtractor, TernaryQuantizer


_EXPERT_FORMAT = "aethercore_v3.ternary_expert"


def _load_torch_payload(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a trusted local torch payload while staying compatible with PyTorch versions."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected expert file to contain a dict, got {type(payload)!r}")
    return payload


def _validate_expert_id(expert_id: str) -> str:
    """Validate and normalize an expert identifier."""

    if not isinstance(expert_id, str):
        raise TypeError("expert_id must be a string")
    normalized = expert_id.strip()
    if not normalized:
        raise ValueError("expert_id must not be empty")
    return normalized


class TernaryExpert(nn.Module):
    """A 1-bit/ternary linear expert with optional residual correction."""

    def __init__(
        self,
        ternary_weight: torch.Tensor,
        scale: torch.Tensor | float = 1.0,
        correction_table: Mapping[str, Any] | None = None,
        bias: torch.Tensor | None = None,
        expert_id: str = "",
        active: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an expert from already-ternarized weights."""

        super().__init__()
        weight = self._validate_ternary_weight(ternary_weight)
        if weight.ndim != 2:
            raise ValueError("TernaryExpert currently supports 2-D linear weights only")

        self.expert_id = str(expert_id)
        self._active = bool(active)
        self.metadata: dict[str, Any] = dict(metadata or {})

        self.register_buffer("ternary_weight", weight.contiguous())
        self.register_buffer("scale", self._validate_scale(scale, weight.shape))
        self.register_buffer("bias", self._validate_bias(bias, weight.shape[0]))

        kind, rank, scalar_bias, u, s, v = self._normalize_correction_table(correction_table, weight.shape)
        self.correction_kind = kind
        self.correction_rank = int(rank)
        self.register_buffer("correction_scalar_bias", scalar_bias)
        self.register_buffer("correction_u", u)
        self.register_buffer("correction_s", s)
        self.register_buffer("correction_v", v)

    @classmethod
    def from_float_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        expert_id: str = "",
        threshold_factor: float = 0.7,
        correction_fraction: float = 0.01,
        use_correction: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TernaryExpert":
        """Quantize a floating point linear weight into a ternary expert."""

        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight must be a torch.Tensor")
        if not torch.is_floating_point(weight):
            raise TypeError("weight must be floating point")
        if weight.ndim != 2:
            raise ValueError("weight must be a 2-D tensor shaped [out_features, in_features]")

        quantizer = TernaryQuantizer(threshold_factor=threshold_factor)
        ternary, scale = quantizer.quantize(weight)
        compressed_weight = ternary.float().mul(scale.float())
        correction_table = None
        if use_correction:
            correction_table = CorrectionTableExtractor(max_fraction=correction_fraction).extract(weight, compressed_weight)

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "source_dtype": str(weight.dtype),
                "threshold_factor": float(threshold_factor),
                "correction_fraction": float(correction_fraction),
            }
        )
        return cls(
            ternary_weight=ternary,
            scale=scale,
            correction_table=correction_table,
            bias=bias,
            expert_id=expert_id,
            metadata=merged_metadata,
        )

    @property
    def in_features(self) -> int:
        """Return the input feature count."""

        return int(self.ternary_weight.shape[1])

    @property
    def out_features(self) -> int:
        """Return the output feature count."""

        return int(self.ternary_weight.shape[0])

    @property
    def active(self) -> bool:
        """Return whether this expert is currently active in memory."""

        return self._active

    @property
    def correction_table(self) -> dict[str, Any]:
        """Return the correction table as a serializable dictionary."""

        table: dict[str, Any] = {
            "format": "aethercore_v3.correction_table",
            "shape": tuple(int(v) for v in self.ternary_weight.shape),
            "kind": self.correction_kind,
            "rank": int(self.correction_rank),
            "scalar_bias": float(self.correction_scalar_bias.float().item()),
        }
        if self.correction_kind == "low_rank" and self.correction_rank > 0:
            table.update(
                {
                    "u": self.correction_u.detach().cpu().to(torch.float16),
                    "s": self.correction_s.detach().cpu().to(torch.float16),
                    "v": self.correction_v.detach().cpu().to(torch.float16),
                }
            )
        return table

    def has_correction(self) -> bool:
        """Return true when the expert has a non-empty correction table."""

        scalar = abs(float(self.correction_scalar_bias.float().item())) > 0.0
        low_rank = self.correction_kind == "low_rank" and self.correction_rank > 0
        return bool(scalar or low_rank)

    def forward(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        """Run a linear projection using ternary weights and optional correction."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim == 0 or int(x.shape[-1]) != self.in_features:
            raise ValueError(f"Expected input last dimension {self.in_features}, got {tuple(x.shape)}")

        compute_x = self._prepare_input(x)
        weight = self.dequantized_weight(apply_correction=False).to(device=compute_x.device, dtype=compute_x.dtype)
        bias = None if self.bias is None else self.bias.to(device=compute_x.device, dtype=compute_x.dtype)
        output = F.linear(compute_x, weight, bias)

        if use_correction and self.has_correction():
            output = self._apply_output_correction(output, compute_x)
        return output

    def sparsity(self) -> float:
        """Return the fraction of zero ternary weights."""

        if self.ternary_weight.numel() == 0:
            return 1.0
        return float((self.ternary_weight == 0).sum().item() / self.ternary_weight.numel())

    def sleep(self) -> None:
        """Mark the expert as inactive without unloading its tensors."""

        self._active = False

    def wake(self) -> None:
        """Mark the expert as active."""

        self._active = True

    def dequantized_weight(self, apply_correction: bool = False) -> torch.Tensor:
        """Return the floating point weight represented by this expert."""

        base = self.ternary_weight.float()
        scale = self.scale.float()
        if scale.ndim == 0 or scale.numel() == 1:
            weight = base.mul(scale.reshape(()))
        elif tuple(scale.shape) == (self.out_features,):
            weight = base.mul(scale.view(-1, 1))
        else:
            weight = base.mul(scale)

        if apply_correction and self.has_correction():
            weight = self._apply_weight_correction(weight)
        return weight

    def save_to_file(self, path: str | Path) -> None:
        """Persist this expert to a torch file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _EXPERT_FORMAT,
            "version": 1,
            "expert_id": self.expert_id,
            "active": self.active,
            "ternary_weight": self.ternary_weight.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "bias": None if self.bias is None else self.bias.detach().cpu(),
            "correction_table": self.correction_table,
            "metadata": dict(self.metadata),
        }
        torch.save(payload, output_path)

    @classmethod
    def load_from_file(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "TernaryExpert":
        """Load a ternary expert from a torch file."""

        input_path = Path(path)
        payload = _load_torch_payload(input_path, map_location=map_location)
        if payload.get("format") != _EXPERT_FORMAT:
            raise ValueError(f"Unsupported expert file format: {payload.get('format')!r}")

        expert = cls(
            ternary_weight=payload["ternary_weight"],
            scale=payload["scale"],
            correction_table=payload.get("correction_table"),
            bias=payload.get("bias"),
            expert_id=str(payload.get("expert_id", "")),
            active=bool(payload.get("active", True)),
            metadata=dict(payload.get("metadata", {})),
        )
        return expert.to(map_location)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Choose a CPU-safe floating point dtype for inference."""

        if not torch.is_floating_point(x):
            return x.float()
        if x.dtype in {torch.float16, torch.bfloat16} and x.device.type == "cpu":
            return x.float()
        return x

    def _apply_output_correction(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project the residual correction into output space."""

        corrected = output
        if self.correction_kind == "low_rank" and self.correction_rank > 0:
            residual = self._low_rank_residual().to(device=x.device, dtype=x.dtype)
            corrected = corrected + F.linear(x, residual)

        scalar = self.correction_scalar_bias.to(device=x.device, dtype=x.dtype)
        if float(scalar.float().item()) != 0.0:
            corrected = corrected + x.sum(dim=-1, keepdim=True).mul(scalar)
        return corrected

    def _apply_weight_correction(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply the residual correction directly to a weight matrix."""

        corrected = weight
        if self.correction_kind == "low_rank" and self.correction_rank > 0:
            corrected = corrected + self._low_rank_residual().to(device=weight.device, dtype=weight.dtype)

        scalar = self.correction_scalar_bias.to(device=weight.device, dtype=weight.dtype)
        if float(scalar.float().item()) != 0.0:
            corrected = corrected + scalar
        return corrected

    def _low_rank_residual(self) -> torch.Tensor:
        """Reconstruct the low-rank residual weight matrix."""

        if self.correction_kind != "low_rank" or self.correction_rank <= 0:
            return torch.zeros_like(self.ternary_weight, dtype=torch.float32)
        return (self.correction_u.float() * self.correction_s.float().unsqueeze(0)).matmul(self.correction_v.float())

    def _validate_ternary_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Validate that a tensor contains only ternary values."""

        if not isinstance(weight, torch.Tensor):
            raise TypeError("ternary_weight must be a torch.Tensor")
        if weight.numel() == 0:
            raise ValueError("ternary_weight must not be empty")
        if not torch.all((weight == -1) | (weight == 0) | (weight == 1)).item():
            raise ValueError("ternary_weight must contain only -1, 0, and +1")
        return weight.detach().to(torch.int8)

    def _validate_scale(self, scale: torch.Tensor | float, weight_shape: torch.Size) -> torch.Tensor:
        """Validate a scalar, per-output, or per-weight scale."""

        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        valid_shapes = {
            (),
            (1,),
            (int(weight_shape[0]),),
            (int(weight_shape[0]), 1),
            tuple(int(v) for v in weight_shape),
        }
        if tuple(scale_tensor.shape) not in valid_shapes:
            raise ValueError(
                "scale must be scalar, [out_features], [out_features, 1], "
                "or the same shape as ternary_weight"
            )
        if not torch.isfinite(scale_tensor).all().item():
            raise ValueError("scale must contain finite values")
        return scale_tensor.contiguous()

    def _validate_bias(self, bias: torch.Tensor | None, out_features: int) -> torch.Tensor | None:
        """Validate an optional linear bias."""

        if bias is None:
            return None
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a torch.Tensor or None")
        if tuple(bias.shape) != (int(out_features),):
            raise ValueError(f"bias must have shape ({out_features},), got {tuple(bias.shape)}")
        if not torch.is_floating_point(bias):
            raise TypeError("bias must be floating point")
        return bias.detach().clone().float().contiguous()

    def _normalize_correction_table(
        self,
        correction_table: Mapping[str, Any] | None,
        weight_shape: torch.Size,
    ) -> tuple[str, int, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Normalize correction-table payloads into module buffers."""

        scalar_bias = torch.tensor(0.0, dtype=torch.float16)
        if correction_table is None:
            return "none", 0, scalar_bias, None, None, None
        if not isinstance(correction_table, Mapping):
            raise TypeError("correction_table must be a mapping or None")

        table_shape = tuple(int(v) for v in correction_table.get("shape", weight_shape))
        expected_shape = tuple(int(v) for v in weight_shape)
        if table_shape != expected_shape:
            raise ValueError(f"correction_table shape {table_shape} does not match weight shape {expected_shape}")

        kind = str(correction_table.get("kind", "scalar"))
        if kind not in {"none", "scalar", "low_rank"}:
            raise ValueError(f"Unsupported correction_table kind: {kind!r}")

        scalar_bias = torch.tensor(float(correction_table.get("scalar_bias", 0.0)), dtype=torch.float16)
        if kind != "low_rank":
            return kind, 0, scalar_bias, None, None, None

        required = {"u", "s", "v"}
        missing = required.difference(correction_table)
        if missing:
            raise ValueError(f"low_rank correction_table missing keys: {sorted(missing)}")

        u = torch.as_tensor(correction_table["u"], dtype=torch.float16).detach().clone().contiguous()
        s = torch.as_tensor(correction_table["s"], dtype=torch.float16).detach().clone().contiguous()
        v = torch.as_tensor(correction_table["v"], dtype=torch.float16).detach().clone().contiguous()
        if u.ndim != 2 or s.ndim != 1 or v.ndim != 2:
            raise ValueError("low_rank correction buffers must have shapes u=[out, rank], s=[rank], v=[rank, in]")

        rank = min(int(correction_table.get("rank", s.numel())), int(u.shape[1]), int(s.numel()), int(v.shape[0]))
        if rank <= 0:
            return "scalar", 0, scalar_bias, None, None, None
        if int(u.shape[0]) != int(weight_shape[0]) or int(v.shape[1]) != int(weight_shape[1]):
            raise ValueError("low_rank correction dimensions do not match ternary_weight")

        return "low_rank", rank, scalar_bias, u[:, :rank], s[:rank], v[:rank, :]


class ExpertPool:
    """Registry for active and sleeping ternary experts."""

    def __init__(self) -> None:
        """Create an empty expert pool."""

        self._experts: dict[str, TernaryExpert] = {}

    def register_expert(self, expert_id: str, expert: TernaryExpert) -> None:
        """Register a new expert under a stable identifier."""

        normalized_id = _validate_expert_id(expert_id)
        if not isinstance(expert, TernaryExpert):
            raise TypeError("expert must be a TernaryExpert")
        if normalized_id in self._experts:
            raise KeyError(f"Expert {normalized_id!r} already registered")
        expert.expert_id = normalized_id
        self._experts[normalized_id] = expert

    def get_expert(self, expert_id: str) -> TernaryExpert:
        """Return an expert by identifier."""

        normalized_id = _validate_expert_id(expert_id)
        try:
            return self._experts[normalized_id]
        except KeyError as exc:
            raise KeyError(f"Expert {normalized_id!r} is not registered") from exc

    def active_experts(self) -> list[str]:
        """Return identifiers for experts currently marked active."""

        return [expert_id for expert_id, expert in self._experts.items() if expert.active]

    def sleeping_experts(self) -> list[str]:
        """Return identifiers for experts currently marked inactive."""

        return [expert_id for expert_id, expert in self._experts.items() if not expert.active]

    def sleep_expert(self, expert_id: str) -> None:
        """Mark one registered expert as sleeping."""

        self.get_expert(expert_id).sleep()

    def wake_expert(self, expert_id: str) -> None:
        """Mark one registered expert as active."""

        self.get_expert(expert_id).wake()

    def unregister_expert(self, expert_id: str) -> TernaryExpert:
        """Remove and return a registered expert."""

        normalized_id = _validate_expert_id(expert_id)
        try:
            return self._experts.pop(normalized_id)
        except KeyError as exc:
            raise KeyError(f"Expert {normalized_id!r} is not registered") from exc

    def __contains__(self, expert_id: object) -> bool:
        """Return true when an expert id is registered."""

        return isinstance(expert_id, str) and expert_id in self._experts

    def __len__(self) -> int:
        """Return the number of registered experts."""

        return len(self._experts)


def _self_test() -> None:
    """Run a small CPU sanity check for ternary experts."""

    torch.manual_seed(11)
    weight = torch.randn(12, 8, dtype=torch.float32) * 0.2
    bias = torch.randn(12, dtype=torch.float32) * 0.05
    expert = TernaryExpert.from_float_weight(weight, bias=bias, expert_id="math.core")
    x = torch.randn(4, 8, dtype=torch.float32)
    output = expert(x)
    uncorrected = expert(x, use_correction=False)

    with tempfile.TemporaryDirectory(prefix="aethercore_expert_") as temp_dir:
        path = Path(temp_dir) / "expert.pt"
        expert.save_to_file(path)
        loaded = TernaryExpert.load_from_file(path)
        loaded_output = loaded(x)

    pool = ExpertPool()
    pool.register_expert("math.core", expert)
    pool.register_expert("code.core", TernaryExpert.from_float_weight(torch.randn(10, 8), expert_id="code.core"))
    pool.sleep_expert("code.core")

    if output.shape != (4, 12):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.allclose(output, loaded_output, atol=1.0e-4, rtol=1.0e-4):
        raise RuntimeError("Loaded expert output does not match saved expert output")

    print("AetherCore ternary expert self-test")
    print(f"  expert id: {expert.expert_id}")
    print(f"  sparsity: {expert.sparsity():.3f}")
    print(f"  correction enabled: {expert.has_correction()}")
    print(f"  correction delta mean: {(output - uncorrected).abs().mean().item():.6f}")
    print(f"  output shape: {tuple(output.shape)}")
    print(f"  active experts: {pool.active_experts()}")
    print(f"  sleeping experts: {pool.sleeping_experts()}")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
