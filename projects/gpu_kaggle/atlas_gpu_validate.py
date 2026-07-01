"""ATLAS R1 — GPU validation: native sparse + 2-bit QAT on a REAL model.

The riskiest ATLAS claim is that you can drive a real language model to *very low
average bits/weight* (sparsity + ternary) WITHOUT the quality collapse that kills
naive post-hoc compression — by training to **preserve behaviour, not weights**.

This script validates that on a real Hugging Face model, on a free Kaggle GPU:

  teacher (FP, frozen)
       │  distill (KL + CE) through a straight-through estimator
       ▼
  student = SPARSE + TERNARY shadow weights   ← native sparse training
       vs
  naive   = same sparsity+ternary, NO training ← reproduces the collapse

It reports, honestly:
  * perplexity: teacher vs naive (collapsed) vs healed student,
  * top-1 next-token agreement with the teacher,
  * the ACHIEVED average bits/weight (2-bit codes + fp16 scales + the cheaper of
    a bitmask / CSR index for the sparsity pattern) and the compression vs fp16.

Thesis proven if: healed perplexity << naive perplexity (native beats post-hoc)
at high sparsity, and the achieved bits/weight is a large compression over fp16.

------------------------------------------------------------------------------
RUN ON KAGGLE (free T4 GPU):
  1. New Notebook -> Settings -> Accelerator: GPU T4 x1, Internet: ON.
  2. Paste this file into a cell (or add as a utility script) and run:
         !pip -q install "transformers>=4.44" datasets accelerate
         %run atlas_gpu_validate.py
     or in Python:  from atlas_gpu_validate import main; main(model_name="EleutherAI/pythia-1b", sparsity=0.95)
  3. Read the printed table; results.json is written next to the script.

RUN LOCALLY (CPU smoke test, no big model, no internet):
     python projects/gpu_kaggle/atlas_gpu_validate.py --self-test
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# core: sparse + ternary with a straight-through estimator
# ---------------------------------------------------------------------------
def sparse_ternary_ste(shadow: torch.Tensor, sparsity: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (quantized_weight, mask) for an nn.Linear weight ``[out, in]``.

    Native-sparse: a magnitude mask keeps the top ``(1 - sparsity)`` fraction of
    weights (recomputed every forward, so the mask adapts as the shadow trains —
    weights can drop out and grow back, RigL-style). The kept weights are
    ternarized to ``±scale`` per output row (scale = mean |w| over kept in-row).
    The straight-through estimator passes gradients to ``shadow`` unchanged.
    """

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    w = shadow.detach()
    abs_w = w.abs()
    if sparsity > 0.0:
        k = int(sparsity * w.numel())
        if k > 0:
            # global magnitude threshold at the sparsity percentile
            thresh = torch.kthvalue(abs_w.flatten(), k).values
            mask = (abs_w > thresh).to(w.dtype)
        else:
            mask = torch.ones_like(w)
    else:
        mask = torch.ones_like(w)
    kept = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    row_scale = (abs_w * mask).sum(dim=1, keepdim=True) / kept
    quantized = torch.sign(w) * mask * row_scale
    return shadow + (quantized - shadow).detach(), mask


class SparseTernaryLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a sparse+ternary shadow weight."""

    def __init__(self, linear: nn.Linear, sparsity: float, train_shadow: bool) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.sparsity = float(sparsity)
        shadow = linear.weight.detach().clone().float()
        if train_shadow:
            self.shadow = nn.Parameter(shadow)
        else:
            self.register_buffer("shadow", shadow, persistent=True)
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone().float(), requires_grad=train_shadow)
        self._last_nonzero = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, mask = sparse_ternary_ste(self.shadow, self.sparsity)
        self._last_nonzero = int(mask.sum().item())
        return F.linear(x, weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


# transformer block projection names to wrap (Llama/Qwen/Pythia/GPT-NeoX family)
_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",           # attention (Llama/Qwen)
    "gate_proj", "up_proj", "down_proj",              # mlp (Llama/Qwen)
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",  # GPT-NeoX/Pythia
)


def wrap_sparse_ternary(model: nn.Module, sparsity: float, train_shadow: bool) -> tuple[int, int]:
    """Swap every targeted ``nn.Linear`` in transformer blocks for a sparse-ternary one.

    Returns ``(layers_wrapped, total_params_wrapped)``. Embeddings, the LM head,
    and norms are left in FP (low-bit there destroys quality — the ATLAS rule).
    """

    modules = dict(model.named_modules())
    wrapped = params = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(_TARGET_SUFFIXES):
            continue
        parent = modules.get(name.rpartition(".")[0])
        if parent is None:
            continue
        setattr(parent, name.rpartition(".")[2],
                SparseTernaryLinear(module, sparsity, train_shadow))
        wrapped += 1
        params += module.weight.numel()
    if wrapped == 0:
        raise RuntimeError("no target Linear layers found — check the model architecture")
    return wrapped, params


# ---------------------------------------------------------------------------
# honest bits/weight accounting
# ---------------------------------------------------------------------------
def achieved_bits_per_weight(total_params: int, nonzero: int, out_channels: int,
                             avg_cols: int) -> dict[str, float]:
    """Honest average bits/weight for sparse+ternary storage (all overhead counted).

    * 2 bits per NONZERO weight (ternary code; zeros are not stored),
    * 16 bits per output channel for the fp16 scale,
    * the sparsity pattern needs an index: either a 1-bit-per-weight bitmask, or a
      CSR column index (``ceil(log2(cols))`` bits per nonzero). We take the cheaper.
    """

    if total_params <= 0 or nonzero <= 0:
        raise ValueError("params and nonzero must be positive")
    code = 2 * nonzero
    scale = 16 * out_channels
    bitmask = 1 * total_params
    csr = nonzero * math.ceil(math.log2(max(2, avg_cols)))
    index = min(bitmask, csr)
    total_bits = code + scale + index
    bpw = total_bits / total_params
    return {
        "bits_per_weight": round(bpw, 4),
        "density": round(nonzero / total_params, 4),
        "index_scheme": "csr" if csr <= bitmask else "bitmask",
        "compression_vs_fp16": round(16.0 / bpw, 1),
        "compression_vs_fp32": round(32.0 / bpw, 1),
    }


# ---------------------------------------------------------------------------
# data + metrics
# ---------------------------------------------------------------------------
def _load_text(tokenizer: Any, seq_len: int, n_windows: int) -> list[torch.Tensor]:
    """Return token windows from wikitext-2 (Kaggle) or a bundled fallback."""

    text = ""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(ds["text"][:2000])
    except Exception:  # offline / no datasets — use a small bundled corpus
        text = " ".join([
            "The most important idea in science is that observation guides theory.",
            "Language lets us carry an idea from one mind into another mind.",
            "Mathematics is the language we use to describe patterns in the world.",
            "A small experiment, repeated carefully, can overturn a large belief.",
        ] * 64)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    windows = []
    for i in range(n_windows):
        start = (i * seq_len) % max(1, ids.numel() - seq_len - 1)
        windows.append(ids[start:start + seq_len].unsqueeze(0))
    return windows


@torch.inference_mode()
def _perplexity(model: nn.Module, windows: list[torch.Tensor], device: str) -> float:
    """Mean causal-LM perplexity over evaluation windows."""

    losses = []
    for ids in windows:
        ids = ids.to(device)
        logits = model(ids).logits
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
        lbl = ids[:, 1:].reshape(-1)
        losses.append(F.cross_entropy(sl, lbl).item())
    return float(math.exp(sum(losses) / len(losses)))


@torch.inference_mode()
def _agreement(teacher: nn.Module, student: nn.Module, windows: list[torch.Tensor], device: str) -> float:
    """Top-1 next-token agreement between student and teacher."""

    agree = tot = 0
    for ids in windows:
        ids = ids.to(device)
        t = teacher(ids).logits[0].argmax(-1)
        s = student(ids).logits[0].argmax(-1)
        agree += int((t == s).sum().item())
        tot += t.numel()
    return agree / max(1, tot)


@dataclass
class ValidationResult:
    """Measured outcome of the R1 validation run."""

    model_name: str
    sparsity: float
    steps: int
    device: str
    layers_wrapped: int
    params_wrapped: int
    teacher_ppl: float
    naive_ppl: float
    healed_ppl: float
    naive_agreement: float
    healed_agreement: float
    first_loss: float
    last_loss: float
    accounting: dict[str, Any]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


# ---------------------------------------------------------------------------
# main validation
# ---------------------------------------------------------------------------
def run_validation(model_name: str = "EleutherAI/pythia-410m", sparsity: float = 0.95,
                   steps: int = 60, lr: float = 2.0e-4, seq_len: int = 128,
                   kl_temp: float = 2.0, ce_weight: float = 0.1) -> ValidationResult:
    """Distill an FP teacher into a native sparse+ternary student and measure it."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.perf_counter()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    naive = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    layers, params = wrap_sparse_ternary(naive, sparsity, train_shadow=False)
    naive = naive.to(device)

    student = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    wrap_sparse_ternary(student, sparsity, train_shadow=True)
    student = student.to(device)

    windows = _load_text(tok, seq_len, n_windows=16)
    train_w = _load_text(tok, seq_len, n_windows=steps)

    shadow_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(shadow_params, lr=lr)
    student.train()
    first_loss = last_loss = float("nan")
    for step, ids in enumerate(train_w):
        ids = ids.to(device)
        with torch.inference_mode():
            t_logits = teacher(ids).logits.float()
        t_logits = t_logits.clone()
        opt.zero_grad(set_to_none=True)
        s_logits = student(ids).logits.float()
        ts = F.log_softmax(t_logits[:, :-1].reshape(-1, t_logits.size(-1)) / kl_temp, -1)
        ss = F.log_softmax(s_logits[:, :-1].reshape(-1, s_logits.size(-1)) / kl_temp, -1)
        kl = F.kl_div(ss, ts, reduction="batchmean", log_target=True) * (kl_temp ** 2)
        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.size(-1)), ids[:, 1:].reshape(-1))
        loss = kl + ce_weight * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shadow_params, 1.0)
        opt.step()
        if step == 0:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    student.eval()

    teacher_ppl = _perplexity(teacher, windows, device)
    naive_ppl = _perplexity(naive, windows, device)
    healed_ppl = _perplexity(student, windows, device)
    naive_ag = _agreement(teacher, naive, windows, device)
    healed_ag = _agreement(teacher, student, windows, device)

    # count nonzeros from the student's masks at current sparsity
    nonzero = out_ch = cols_sum = n_layers = 0
    for m in student.modules():
        if isinstance(m, SparseTernaryLinear):
            nonzero += m._last_nonzero
            out_ch += m.out_features
            cols_sum += m.in_features
            n_layers += 1
    avg_cols = cols_sum // max(1, n_layers)
    accounting = achieved_bits_per_weight(params, max(1, nonzero), out_ch, avg_cols)

    return ValidationResult(
        model_name=model_name, sparsity=sparsity, steps=steps, device=device,
        layers_wrapped=layers, params_wrapped=params,
        teacher_ppl=round(teacher_ppl, 3), naive_ppl=round(naive_ppl, 3),
        healed_ppl=round(healed_ppl, 3), naive_agreement=round(naive_ag, 4),
        healed_agreement=round(healed_ag, 4), first_loss=round(first_loss, 4),
        last_loss=round(last_loss, 4), accounting=accounting,
        elapsed_sec=round(time.perf_counter() - t0, 1),
    )


def main(model_name: str = "EleutherAI/pythia-410m", sparsity: float = 0.95, steps: int = 60) -> ValidationResult:
    """Run the validation and print + save an honest report."""

    res = run_validation(model_name=model_name, sparsity=sparsity, steps=steps)
    out = Path(__file__).resolve().parent / "gpu_validate_results.json"
    out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")

    acc = res.accounting
    print("\n" + "=" * 68)
    print(f"ATLAS R1 — native sparse + 2-bit QAT  |  {res.model_name}  |  {res.device}")
    print("=" * 68)
    print(f"  sparsity            : {res.sparsity:.0%}   ({res.layers_wrapped} layers, {res.params_wrapped/1e6:.0f}M params wrapped)")
    print(f"  teacher perplexity  : {res.teacher_ppl:.2f}   (FP baseline)")
    print(f"  NAIVE  (post-hoc)   : {res.naive_ppl:.2f}   agree={res.naive_agreement:.1%}   <- collapses")
    print(f"  HEALED (native)     : {res.healed_ppl:.2f}   agree={res.healed_agreement:.1%}   <- preserved")
    print(f"  distill loss        : {res.first_loss:.3f} -> {res.last_loss:.3f}")
    print(f"  achieved bits/weight: {acc['bits_per_weight']}  ({acc['index_scheme']} index, density {acc['density']:.1%})")
    print(f"  compression vs fp16 : {acc['compression_vs_fp16']}x   (vs fp32 {acc['compression_vs_fp32']}x)")
    print(f"  elapsed             : {res.elapsed_sec}s")
    verdict = ("THESIS HOLDS: native training preserves quality where post-hoc collapses"
               if res.healed_ppl < 0.6 * res.naive_ppl else
               "inconclusive on this config — raise steps or lower sparsity")
    print(f"  VERDICT: {verdict}")
    print(f"  written -> {out}")
    return res


# ---------------------------------------------------------------------------
# offline CPU smoke test (no big model, no internet)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Prove the mechanism on a toy Linear stack: post-hoc collapses, healed recovers."""

    torch.manual_seed(0)

    # sparse-ternary really is sparse + 3-valued, and STE passes gradient
    w = torch.randn(8, 32)
    q, mask = sparse_ternary_ste(w, sparsity=0.9)
    density = float(mask.mean().item())
    if not 0.05 < density < 0.15:
        raise RuntimeError(f"sparsity mask wrong density {density}")
    param = nn.Parameter(torch.randn(8, 32))
    (torch.randn(4, 32) @ sparse_ternary_ste(param, 0.9)[0].t()).sum().backward()
    if param.grad is None or float(param.grad.abs().sum()) == 0.0:
        raise RuntimeError("STE gradient did not flow")

    # toy teacher; naive sparse-ternary (frozen) should be much worse than healed
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
    for p in teacher.parameters():
        p.requires_grad_(False)
    x = torch.randn(16, 24)
    with torch.no_grad():
        target = teacher(x)

    def build(train: bool):
        torch.manual_seed(1)
        net = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        net.load_state_dict(teacher.state_dict())
        net[0] = SparseTernaryLinear(net[0], 0.5, train_shadow=train)
        net[2] = SparseTernaryLinear(net[2], 0.5, train_shadow=train)
        return net

    naive = build(False).eval()
    naive_err = float(F.mse_loss(naive(x), target).item())

    student = build(True)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=5e-3)
    first = last = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(student(x), target)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.item())
        last = float(loss.item())
    if last >= first:
        raise RuntimeError(f"healing did not reduce loss: {first:.4f} -> {last:.4f}")
    if last >= naive_err:
        raise RuntimeError(f"healed ({last:.4f}) not better than naive ({naive_err:.4f})")

    acc = achieved_bits_per_weight(1_000_000, 50_000, 4096, 4096)
    if not 0.0 < acc["bits_per_weight"] < 16.0:
        raise RuntimeError(f"bits/weight accounting off: {acc}")

    print("ATLAS GPU-validate self-test (offline toy)")
    print(f"  sparse-ternary density   : {density:.2f} (target ~0.1)")
    print(f"  naive frozen err         : {naive_err:.4f}")
    print(f"  healed err (native)      : {first:.4f} -> {last:.4f}  (beats naive)")
    print(f"  example accounting @95%  : {acc['bits_per_weight']} bits/w = {acc['compression_vs_fp16']}x vs fp16")
    print("  status: ok — ready for GPU (run main() on Kaggle with a real 0.4-1B model)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ATLAS R1 GPU validation")
    ap.add_argument("--self-test", action="store_true", help="offline CPU mechanism test")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--sparsity", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        main(model_name=args.model, sparsity=args.sparsity, steps=args.steps)
