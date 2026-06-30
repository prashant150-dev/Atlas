"""Interactive side-by-side: Normal GPT-2 (FP32) vs our VQ-compressed GPT-2.

HONEST scoping (read this):
- "Our architecture" here = GPT-2 with our P-A SIZE lever applied: vector
  quantization (d=4, K=256, ~2 bits/weight) on the transformer block weights,
  optionally healed (--heal). This is a REAL, runnable artifact.
- It is NOT a "VQ-MoE language model": we never trained an MoE GPT-2 for English
  (MoE was validated separately on synthetic tasks). So this tool does not pretend
  to run one.
- Metrics are honest: DISK is genuinely smaller (the real win). RAM and tok/s are
  ~the same because the VQ weights are dequantised to fp32 at runtime and this CPU
  has no low-bit kernel (Phase B). We show what RAM *would* be with a packed kernel,
  clearly labelled "if-kernel".

Usage (run it yourself for live chat — type `!` prefix in this session):
    !python projects/day12_intelligence/manual_test.py
    !python projects/day12_intelligence/manual_test.py --heal     # heal the VQ model (~2 min)
    python projects/day12_intelligence/manual_test.py --demo      # non-interactive sanity run
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
sys.path.insert(0, str(_HERE.parent / "day4_healing_ceiling"))
from vq_vs_scalar import vector_quant  # type: ignore  # noqa: E402

_MODEL = "models/gpt2"
_WRAP = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
_VQ_BPW = 2.0156      # measured bits/weight for VQ d4K256 (codebook+index overhead in)
MAX_NEW = 40


def _fresh():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(_MODEL, local_files_only=True).eval()


def _disk_bytes(model, vq=False):
    fp = 0
    for name, p in model.named_parameters():
        if vq and name.endswith(_WRAP):
            fp += p.numel() * _VQ_BPW / 8.0      # packed ~2-bit
        else:
            fp += p.numel() * 4                  # fp32
    return fp


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def _apply_vq(model):
    for name, p in model.named_parameters():
        if name.endswith(_WRAP):
            W = p.detach().cpu().float().numpy()
            recon, _ = vector_quant(W, 4, 256, seed=0)
            p.data.copy_(torch.from_numpy(recon.astype(np.float32)).reshape(p.shape))


def _heal(model, teacher, tok, steps=40):
    sys.path.insert(0, str(_HERE.parent / "day6_vector_quant"))
    from p3_vq_heal import VQConv1D, wrap_vq_student  # noqa
    from src.compression.healing_qat import _distillation_loss
    from corpus import TRAIN_TEXT
    # rebuild as trainable VQ student
    student = _fresh()
    wrap_vq_student(student, 4, 256, 0)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-4)
    wins = []
    for t in TRAIN_TEXT:
        ids = tok(t, return_tensors="pt").input_ids[0]
        if ids.numel() < 64:
            ids = ids.repeat((64 // max(1, ids.numel())) + 1)
        wins.append(ids[:64].unsqueeze(0))
    student.train()
    for s in range(steps):
        b = wins[s % len(wins)]
        with torch.inference_mode():
            tl = teacher(b).logits
        tl = tl.clone(); opt.zero_grad(set_to_none=True)
        _distillation_loss(student(b).logits, tl, b, 2.0, 0.1).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    student.eval()
    return student


@torch.inference_mode()
def _gen(model, tok, prompt):
    inp = tok(prompt, return_tensors="pt")
    t = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t
    n = out.shape[1] - inp.input_ids.shape[1]
    txt = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return txt.strip(), n / dt if dt else 0.0


def _panel(title, text, toks, disk_mb, ram_note):
    print(f"\n  +-- {title} " + "-" * max(2, 46 - len(title)))
    for line in (text or "(empty)").split("\n"):
        while len(line) > 60:
            print(f"  | {line[:60]}"); line = line[60:]
        print(f"  | {line}")
    print(f"  |  speed : {toks:5.1f} tok/s")
    print(f"  |  disk  : {disk_mb:6.1f} MB")
    print(f"  +- RAM   : {ram_note}")


def main():
    heal = "--heal" in sys.argv
    demo = "--demo" in sys.argv
    torch.set_num_threads(4)

    print("loading FP32 GPT-2 ...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    fp = _fresh()
    fp_disk = _disk_bytes(fp) / 1e6

    if heal:
        print("healing VQ model (~2 min, 40 steps) ...", flush=True)
        vq = _heal(fp, fp, tok, steps=40)
        vq_label = "VQ+Heal GPT-2 (~2-bit, our size lever)"
    else:
        print("compressing GPT-2 -> VQ ~2-bit (~2 min) ...", flush=True)
        vq = _fresh(); _apply_vq(vq)
        vq_label = "VQ GPT-2 (~2-bit post-hoc, our size lever)"
    vq_disk = _disk_bytes(vq, vq=True) / 1e6

    rss = _rss_mb()
    fp_ram = f"{fp.num_parameters()*4/1e6:.0f} MB fp32 weights"
    vq_ram = f"{fp.num_parameters()*4/1e6:.0f} MB fp32 now (if-kernel: ~{vq_disk:.0f} MB)"

    print("\n" + "=" * 64)
    print(" AetherCore live: Normal GPT-2  vs  our VQ-compressed GPT-2")
    print(f" disk: FP {fp_disk:.0f} MB  ->  VQ {vq_disk:.0f} MB  ({fp_disk/vq_disk:.1f}x smaller)")
    print(f" honest: RAM/speed ~equal on this CPU (no low-bit kernel); win is DISK.")
    if rss:
        print(f" process RSS now (both models loaded): {rss:.0f} MB")
    print("=" * 64)

    prompts = (["The meaning of life is", "Once upon a time"] if demo else None)

    def handle(p):
        ft, fs = _gen(fp, tok, p)
        vt, vs = _gen(vq, tok, p)
        print(f"\nPROMPT: {p}")
        _panel("Normal GPT-2 (FP32)", ft, fs, fp_disk, fp_ram)
        _panel(vq_label, vt, vs, vq_disk, vq_ram)

    if demo:
        for p in prompts:
            handle(p)
        return

    print("\nType a prompt (or 'exit'):")
    while True:
        try:
            p = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break
        if p.lower() in {"exit", "quit", "q"}:
            print("bye!"); break
        if not p:
            continue
        handle(p)


if __name__ == "__main__":
    main()
