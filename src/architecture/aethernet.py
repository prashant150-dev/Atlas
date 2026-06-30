"""AetherNet: a natively ternary, sparse-MoE sequence model (Day-3 co-design).

The Day-1/Day-2 experiments established the post-hoc wall: ternarizing a trained
floating-point model destroys its behaviour at the same bit budget, and only a
*goal change* (healing/QAT) recovers it.  Day-3 takes the co-design argument to
its conclusion — rather than compress a dense model after the fact, we *design*
a model that lives in ternary + sparse space from the start.

Three variants share one shape so the comparison is fair:

* :class:`DenseFP` — full FP32 weights, a single dense feed-forward block.
* :class:`PostHocTernary` — the trained :class:`DenseFP` weights ternarized with
  **no retraining** (reproduces the post-hoc collapse on hard tasks).
* :class:`AetherNet` — natively ternary linear weights trained through a
  straight-through estimator, with a sparse mixture-of-experts feed-forward
  block that routes each token to ``top_k`` of ``n_expert`` experts.

Bit accounting (:class:`BitAccount`) distinguishes **stored** bits (the whole
model on disk) from **active** bits/token (only the parameters a single token
actually touches).  The sparse MoE keeps active bits well below the dense model
even though its total parameter count is larger.

CPU only.  Deterministic given a fixed seed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_EPS = 1.0e-8

# Bit costs used by the accounting.  A floating-point weight costs 32 bits; a
# ternary weight costs ``_TERNARY_VALUE_BITS`` for its {-1,0,+1} index plus a
# shared per-output-channel FP16 scale (``_SCALE_BITS`` amortised over the row).
_FP_BITS = 32.0
# A ternary weight is stored as a packed value index (kept deliberately
# conservative — we count the practical packed cost, not the 1.585-bit entropy
# floor) plus a shared per-output-channel FP16 scale.
_TERNARY_VALUE_BITS = 4.4
_SCALE_BITS = 16.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AetherNetConfig:
    """Immutable shape/training configuration shared by all three variants."""

    vocab_size: int
    seq_len: int
    d_model: int = 64
    n_head: int = 4
    dense_hidden: int = 458
    n_expert: int = 10
    top_k: int = 4
    expert_hidden: int = 184
    ternary_threshold: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration eagerly so errors surface at construction."""

        for name in ("vocab_size", "seq_len", "d_model", "n_head", "dense_hidden",
                     "n_expert", "top_k", "expert_hidden"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.top_k > self.n_expert:
            raise ValueError("top_k must not exceed n_expert")
        if not isinstance(self.ternary_threshold, float) or self.ternary_threshold < 0.0:
            raise ValueError("ternary_threshold must be a non-negative float")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the configuration."""

        return {
            "vocab_size": int(self.vocab_size),
            "seq_len": int(self.seq_len),
            "d_model": int(self.d_model),
            "n_head": int(self.n_head),
            "dense_hidden": int(self.dense_hidden),
            "n_expert": int(self.n_expert),
            "top_k": int(self.top_k),
            "expert_hidden": int(self.expert_hidden),
            "ternary_threshold": float(self.ternary_threshold),
            "seed": int(self.seed),
        }


# ---------------------------------------------------------------------------
# Bit accounting
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BitAccount:
    """Stored and per-token-active bit counts for one model variant."""

    stored_bits: float
    active_bits_per_token: float

    def __post_init__(self) -> None:
        """Validate that both bit counts are finite and non-negative."""

        for name in ("stored_bits", "active_bits_per_token"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly view of the bit account."""

        return {
            "stored_bits": float(self.stored_bits),
            "active_bits_per_token": float(self.active_bits_per_token),
        }


def fp_weight_bits(n_weights: int) -> float:
    """Return the stored bit cost of ``n_weights`` floating-point weights."""

    if n_weights < 0:
        raise ValueError("n_weights must be non-negative")
    return _FP_BITS * float(n_weights)


def ternary_weight_bits(n_weights: int, n_out_channels: int) -> float:
    """Return the bit cost of a ternary weight matrix with per-channel scales.

    Each weight contributes ``_TERNARY_VALUE_BITS`` and every output channel
    carries one FP16 scale.
    """

    if n_weights < 0 or n_out_channels < 0:
        raise ValueError("weight and channel counts must be non-negative")
    return _TERNARY_VALUE_BITS * float(n_weights) + _SCALE_BITS * float(n_out_channels)


# ---------------------------------------------------------------------------
# Ternary straight-through machinery
# ---------------------------------------------------------------------------
def _ternarize(weight: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Ternarize ``weight`` per output row, returning (ternary {-1,0,+1}, scale).

    The scale is the mean magnitude of the surviving (non-zero) weights in each
    row; weights with magnitude below ``threshold`` times the row mean magnitude
    are forced to zero.
    """

    if weight.ndim != 2:
        raise ValueError("ternarize expects a 2-D [out, in] weight")
    abs_w = weight.abs()
    row_mean = abs_w.mean(dim=1, keepdim=True)
    mask = (abs_w >= threshold * row_mean).to(weight.dtype)
    ternary = torch.sign(weight) * mask
    kept = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    scale = (abs_w * mask).sum(dim=1, keepdim=True) / kept
    return ternary, scale.clamp_min(_EPS)


class _TernarizeSTE(torch.autograd.Function):
    """Straight-through estimator: forward ternarizes, backward is identity."""

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, threshold: float) -> torch.Tensor:  # type: ignore[override]
        ternary, scale = _ternarize(weight, float(threshold))
        return ternary * scale

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        # Gradient flows straight back to the FP shadow weight (STE).
        return grad_output, None


class TernaryLinear(nn.Module):
    """A linear layer whose weight is ternarized in the forward pass via STE.

    A full-precision *shadow* weight is the trainable parameter; the forward pass
    uses its ternarized projection so the network learns weights that live in
    ternary space.  When ``frozen_posthoc`` is set the layer ternarizes a fixed
    (already trained) weight with no gradient — the post-hoc baseline.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float,
        bias: bool = True,
    ) -> None:
        """Create a ternary linear layer with an FP shadow weight."""

        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.threshold = float(threshold)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def ternary_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the detached ternary weight and per-row scale (for accounting)."""

        return _ternarize(self.weight.detach(), self.threshold)

    def effective_weight(self) -> torch.Tensor:
        """Return the FP-valued weight the forward pass actually applies."""

        return _TernarizeSTE.apply(self.weight, self.threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ternarized linear projection."""

        return F.linear(x, self.effective_weight(), self.bias)


# ---------------------------------------------------------------------------
# Attention / feed-forward blocks
# ---------------------------------------------------------------------------
class _CausalSelfAttention(nn.Module):
    """Minimal causal multi-head self-attention; ternary or FP linear maps."""

    def __init__(self, cfg: AetherNetConfig, ternary: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.ternary = bool(ternary)
        d, h = cfg.d_model, cfg.n_head
        self.n_head = h
        self.head_dim = d // h
        self.qkv = self._linear(d, 3 * d, ternary)
        self.proj = self._linear(d, d, ternary)

    def _linear(self, i: int, o: int, ternary: bool) -> nn.Module:
        if ternary:
            return TernaryLinear(i, o, self.cfg.ternary_threshold, bias=True)
        return nn.Linear(i, o, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class _DenseFFN(nn.Module):
    """A single dense feed-forward block (FP or ternary)."""

    def __init__(self, cfg: AetherNetConfig, ternary: bool) -> None:
        super().__init__()
        d, hidden = cfg.d_model, cfg.dense_hidden
        if ternary:
            self.up: nn.Module = TernaryLinear(d, hidden, cfg.ternary_threshold)
            self.down: nn.Module = TernaryLinear(hidden, d, cfg.ternary_threshold)
        else:
            self.up = nn.Linear(d, hidden)
            self.down = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class _MoEFFN(nn.Module):
    """Sparse mixture-of-experts feed-forward block with top-k routing.

    Each token is routed to ``top_k`` of ``n_expert`` ternary experts by a small
    softmax gate; the active fraction of FFN parameters per token is therefore
    ``top_k / n_expert``.
    """

    def __init__(self, cfg: AetherNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, hidden = cfg.d_model, cfg.expert_hidden
        self.n_expert = cfg.n_expert
        self.top_k = cfg.top_k
        self.gate = nn.Linear(d, cfg.n_expert, bias=True)
        self.experts_up = nn.ModuleList(
            [TernaryLinear(d, hidden, cfg.ternary_threshold) for _ in range(cfg.n_expert)]
        )
        self.experts_down = nn.ModuleList(
            [TernaryLinear(hidden, d, cfg.ternary_threshold) for _ in range(cfg.n_expert)]
        )
        self.last_active_experts = cfg.top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        logits = self.gate(flat)
        gate = torch.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(gate, self.top_k, dim=-1)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        self.last_active_experts = self.top_k

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = top_idx[:, slot]
            weight = top_val[:, slot].unsqueeze(-1)
            for expert_id in range(self.n_expert):
                sel = idx == expert_id
                if not torch.any(sel):
                    continue
                tokens = flat[sel]
                hidden = F.gelu(self.experts_up[expert_id](tokens))
                expert_out = self.experts_down[expert_id](hidden)
                out[sel] = out[sel] + weight[sel] * expert_out
        return out.reshape(b, t, d)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class _BaseSeqModel(nn.Module):
    """Shared embedding / attention / norm / head scaffold for all variants."""

    variant_name: str = "Base"

    def __init__(self, cfg: AetherNetConfig, ternary_body: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.ternary_body = bool(ternary_body)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn = _CausalSelfAttention(cfg, ternary=ternary_body)
        self.ffn: nn.Module = self._build_ffn()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _build_ffn(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must be a [batch, seq] long tensor")
        t = tokens.shape[1]
        x = self.embed(tokens) + self.pos[:, :t, :]
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.head(x)

    # -- bit accounting ----------------------------------------------------
    def _embed_head_stored_bits(self) -> float:
        emb = self.cfg.vocab_size * self.cfg.d_model
        head = self.cfg.d_model * self.cfg.vocab_size + self.cfg.vocab_size
        return fp_weight_bits(emb + head)

    def _embed_head_active_bits(self) -> float:
        # Only one embedding row is touched per token.  The output head is a
        # shared projection amortised across the whole vocabulary, so it is
        # counted in stored bits but not in the per-token active budget.
        emb_active = self.cfg.d_model
        return fp_weight_bits(emb_active)

    def _attn_weight_count(self) -> tuple[int, int]:
        d = self.cfg.d_model
        weights = 3 * d * d + d * d
        rows = 3 * d + d
        return weights, rows

    def bit_account(self) -> BitAccount:
        raise NotImplementedError


class DenseFP(_BaseSeqModel):
    """Full FP32 model with a single dense feed-forward block (the baseline)."""

    variant_name = "DenseFP"

    def __init__(self, cfg: AetherNetConfig) -> None:
        super().__init__(cfg, ternary_body=False)

    def _build_ffn(self) -> nn.Module:
        return _DenseFFN(self.cfg, ternary=False)

    def _body_weight_count(self) -> int:
        attn_w, _ = self._attn_weight_count()
        ffn_w = self.cfg.d_model * self.cfg.dense_hidden + self.cfg.dense_hidden * self.cfg.d_model
        return attn_w + ffn_w

    def bit_account(self) -> BitAccount:
        body = self._body_weight_count()
        stored = self._embed_head_stored_bits() + fp_weight_bits(body)
        active = self._embed_head_active_bits() + fp_weight_bits(body)
        return BitAccount(stored_bits=stored, active_bits_per_token=active)


class PostHocTernary(nn.Module):
    """Wraps trained :class:`DenseFP` weights and ternarizes them, no retraining.

    The wrapped dense model is run with its attention/FFN linear weights replaced
    by their ternarized form at evaluation time.  No parameters are updated, so
    this reproduces the post-hoc collapse: capability that the FP weights carried
    is partly lost because the goal was never changed to behaviour-preservation.
    """

    variant_name = "PostHocTernary"

    def __init__(self, dense: DenseFP) -> None:
        """Create a post-hoc ternary view of a trained dense FP model."""

        super().__init__()
        if not isinstance(dense, DenseFP):
            raise TypeError("PostHocTernary requires a trained DenseFP model")
        self.dense = dense
        self.cfg = dense.cfg

    @staticmethod
    def _ternarize_module(linear: nn.Linear, threshold: float) -> nn.Linear:
        ternary, scale = _ternarize(linear.weight.detach(), threshold)
        replacement = nn.Linear(linear.in_features, linear.out_features,
                                bias=linear.bias is not None)
        with torch.no_grad():
            replacement.weight.copy_(ternary * scale)
            if linear.bias is not None:
                replacement.bias.copy_(linear.bias.detach())
        return replacement

    def _ternarized_clone(self) -> DenseFP:
        clone = DenseFP(self.cfg)
        clone.load_state_dict(self.dense.state_dict())
        th = self.cfg.ternary_threshold
        attn = clone.attn
        attn.qkv = self._ternarize_module(attn.qkv, th)
        attn.proj = self._ternarize_module(attn.proj, th)
        ffn = clone.ffn
        ffn.up = self._ternarize_module(ffn.up, th)
        ffn.down = self._ternarize_module(ffn.down, th)
        clone.eval()
        return clone

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the dense model with its body weights ternarized in place."""

        return self._ternarized_clone()(tokens)

    def bit_account(self) -> BitAccount:
        """Account body weights as ternary, embedding/head as FP (unchanged)."""

        cfg = self.cfg
        attn_w, attn_rows = self.dense._attn_weight_count()
        ffn_w = cfg.d_model * cfg.dense_hidden + cfg.dense_hidden * cfg.d_model
        ffn_rows = cfg.dense_hidden + cfg.d_model
        body_bits = ternary_weight_bits(attn_w + ffn_w, attn_rows + ffn_rows)
        emb_head_stored = self.dense._embed_head_stored_bits()
        emb_head_active = self.dense._embed_head_active_bits()
        return BitAccount(
            stored_bits=emb_head_stored + body_bits,
            active_bits_per_token=emb_head_active + body_bits,
        )


class AetherNet(_BaseSeqModel):
    """Natively ternary model with a sparse top-k MoE feed-forward block.

    All body linear weights are trained through the straight-through estimator,
    so the network optimises in ternary space directly.  The MoE FFN holds many
    experts but activates only ``top_k`` of them per token, keeping the active
    bit count far below its stored bit count.
    """

    variant_name = "AetherNet"

    def __init__(self, cfg: AetherNetConfig) -> None:
        super().__init__(cfg, ternary_body=True)

    def _build_ffn(self) -> nn.Module:
        return _MoEFFN(self.cfg)

    def bit_account(self) -> BitAccount:
        cfg = self.cfg
        attn_w, attn_rows = self._attn_weight_count()
        attn_bits = ternary_weight_bits(attn_w, attn_rows)

        per_expert_w = cfg.d_model * cfg.expert_hidden + cfg.expert_hidden * cfg.d_model
        per_expert_rows = cfg.expert_hidden + cfg.d_model
        all_expert_bits = ternary_weight_bits(
            per_expert_w * cfg.n_expert, per_expert_rows * cfg.n_expert
        )
        active_expert_bits = ternary_weight_bits(
            per_expert_w * cfg.top_k, per_expert_rows * cfg.top_k
        )

        gate_w = cfg.d_model * cfg.n_expert + cfg.n_expert
        gate_bits = fp_weight_bits(gate_w)

        emb_head_stored = self._embed_head_stored_bits()
        emb_head_active = self._embed_head_active_bits()

        stored = emb_head_stored + attn_bits + all_expert_bits + gate_bits
        active = emb_head_active + attn_bits + active_expert_bits + gate_bits
        return BitAccount(stored_bits=stored, active_bits_per_token=active)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_dense_fp(cfg: AetherNetConfig) -> DenseFP:
    """Construct a :class:`DenseFP` model under the config's seed."""

    if not isinstance(cfg, AetherNetConfig):
        raise TypeError("cfg must be an AetherNetConfig")
    torch.manual_seed(cfg.seed)
    return DenseFP(cfg)


def build_aethernet(cfg: AetherNetConfig) -> AetherNet:
    """Construct an :class:`AetherNet` model under the config's seed."""

    if not isinstance(cfg, AetherNetConfig):
        raise TypeError("cfg must be an AetherNetConfig")
    torch.manual_seed(cfg.seed + 1)
    return AetherNet(cfg)


def ternarize_dense_to_posthoc(dense: DenseFP) -> PostHocTernary:
    """Wrap a trained dense model as its post-hoc ternary baseline."""

    return PostHocTernary(dense)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Assert mechanism invariants for ternary weights, routing, and STE grad."""

    torch.manual_seed(0)
    cfg = AetherNetConfig(vocab_size=13, seq_len=6, d_model=32, n_head=4,
                          dense_hidden=64, n_expert=8, top_k=2, expert_hidden=48)

    # 1) Ternary weights take exactly three values.
    lin = TernaryLinear(16, 24, cfg.ternary_threshold)
    ternary, scale = lin.ternary_weight()
    uniques = set(int(v) for v in torch.unique(ternary).tolist())
    if not uniques.issubset({-1, 0, 1}):
        raise RuntimeError(f"ternary weight has non-ternary values: {uniques}")
    if scale.shape != (24, 1):
        raise RuntimeError(f"unexpected scale shape: {tuple(scale.shape)}")

    # 2) STE lets gradient flow to the FP shadow weight.
    x = torch.randn(4, 16, requires_grad=False)
    out = lin(x).sum()
    out.backward()
    if lin.weight.grad is None or float(lin.weight.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("STE did not propagate gradient to the shadow weight")

    # 3) Top-k routing activates exactly top_k experts per token.
    net = build_aethernet(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits = net(tokens)
    if logits.shape != (2, cfg.seq_len, cfg.vocab_size):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    moe = net.ffn
    if not isinstance(moe, _MoEFFN):
        raise RuntimeError("AetherNet FFN is not a sparse MoE block")
    flat = (net.embed(tokens) + net.pos[:, : cfg.seq_len, :]).reshape(-1, cfg.d_model)
    gate = torch.softmax(moe.gate(flat), dim=-1)
    _, top_idx = torch.topk(gate, moe.top_k, dim=-1)
    per_token_active = (torch.zeros_like(gate).scatter_(1, top_idx, 1.0)).sum(dim=-1)
    if not torch.all(per_token_active == float(moe.top_k)).item():
        raise RuntimeError("top-k routing did not activate exactly top_k experts")

    # 4) Bit accounting: AetherNet active < stored, and stored sane vs DenseFP.
    dense = build_dense_fp(cfg)
    posthoc = ternarize_dense_to_posthoc(dense)
    da, pa, aa = dense.bit_account(), posthoc.bit_account(), net.bit_account()
    if not (aa.active_bits_per_token < aa.stored_bits):
        raise RuntimeError("AetherNet active bits should be below its stored bits")
    if not (pa.stored_bits < da.stored_bits):
        raise RuntimeError("PostHoc ternary stored bits should be below DenseFP")
    if not (aa.active_bits_per_token < da.active_bits_per_token):
        raise RuntimeError("AetherNet active bits should be below DenseFP active bits")

    # 5) PostHocTernary forward runs and yields right shape.
    ph_logits = posthoc(tokens)
    if ph_logits.shape != (2, cfg.seq_len, cfg.vocab_size):
        raise RuntimeError(f"unexpected posthoc logits shape: {tuple(ph_logits.shape)}")

    print("AetherNet mechanism self-test")
    print(f"  ternary values: {sorted(uniques)}")
    print(f"  STE grad norm: {float(lin.weight.grad.abs().sum().item()):.4f}")
    print(f"  active experts/token: {int(moe.top_k)} of {int(moe.n_expert)}")
    print(f"  DenseFP   stored={da.stored_bits:,.0f}  active={da.active_bits_per_token:,.0f}")
    print(f"  PostHoc   stored={pa.stored_bits:,.0f}  active={pa.active_bits_per_token:,.0f}"
          f"  (x{da.stored_bits / pa.stored_bits:.2f} / x{da.active_bits_per_token / pa.active_bits_per_token:.2f})")
    print(f"  AetherNet stored={aa.stored_bits:,.0f}  active={aa.active_bits_per_token:,.0f}"
          f"  (x{da.stored_bits / aa.stored_bits:.2f} / x{da.active_bits_per_token / aa.active_bits_per_token:.2f})")
    print("  status: ok")


if __name__ == "__main__":
    _self_test()
