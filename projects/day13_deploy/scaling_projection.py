"""HONEST projections: our 2-bit (VQ + C++ kernel) vs fp16/fp32 across model sizes.

NOT benchmarks — we cannot run these on an 8 GB CPU. These are extrapolations from
measured constants:
  - our recipe ~2.02 bits/weight (P-A), native ternary ~1.58 (BitNet);
  - a real C++ low-bit kernel keeps weights PACKED in RAM (R7: 8x less than fp16)
    and makes decode bandwidth-bound on the packed bytes;
  - decode tok/s ~= efficiency * mem_bandwidth / bytes_read_per_token (dense).
Every number is an ESTIMATE.

Run: python projects/day13_deploy/scaling_projection.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "scaling_projection.json"

PARAMS = {"1B": 1e9, "7B": 7e9, "14B": 14e9, "50B": 50e9, "100B": 100e9, "400B": 400e9}
BITS = {"fp32": 32.0, "fp16": 16.0, "ours_2bit": 2.02, "ternary_1.58": 1.58}
EFF = 0.5            # realistic fraction of peak bandwidth a good kernel reaches

# (usable_memory_GB, bandwidth_GB/s)
HW = {
    "This PC (8GB DDR3)":      (6.5,   18.0),
    "Modern PC (64GB DDR5)":   (56.0,  80.0),
    "RTX 4090 (24GB)":         (22.0,  1008.0),
    "H100 (80GB)":             (75.0,  3350.0),
}


def size_gb(p, bits):
    return p * (bits / 8) / 1e9


def tok_s(p, bits, bw_gbs):
    bytes_per_token = p * (bits / 8)
    return EFF * (bw_gbs * 1e9) / bytes_per_token


def main():
    rows = []
    print("=" * 92)
    print("HONEST PROJECTIONS (estimates) — our 2-bit (VQ + C++ kernel) vs fp16")
    print("=" * 92)
    for name, p in PARAMS.items():
        fp16 = size_gb(p, 16)
        ours = size_gb(p, BITS["ours_2bit"])
        print(f"\n### {name} params  |  fp16 {fp16:.0f} GB   ->   ours 2-bit {ours:.1f} GB   ({fp16/ours:.0f}x smaller RAM)")
        entry = {"model": name, "fp16_GB": round(fp16, 1), "ours_2bit_GB": round(ours, 1),
                 "ram_shrink_x": round(fp16/ours, 1), "hardware": {}}
        for hw, (mem, bw) in HW.items():
            fits16 = fp16 <= mem * 0.7
            fits2 = ours <= mem * 0.7
            s16 = tok_s(p, 16, bw) if fits16 else None
            s2 = tok_s(p, BITS["ours_2bit"], bw) if fits2 else None
            f16 = f"{s16:6.0f} t/s" if fits16 else "  no-fit"
            f2 = f"{s2:6.0f} t/s" if fits2 else "  no-fit"
            print(f"    {hw:24s} | fp16: {f16}   | ours-2bit: {f2}")
            entry["hardware"][hw] = {"fp16_fits": fits16, "fp16_tok_s": round(s16,1) if s16 else None,
                                     "ours_fits": fits2, "ours_tok_s": round(s2,1) if s2 else None}
        rows.append(entry)

    print("\n" + "=" * 92)
    print("INTELLIGENCE (capability retained vs fp16) — honest band from our + BitNet evidence:")
    print("  native low-bit / well-healed : ~95-99% of fp16 quality (BitNet b1.58 ~= fp at 1.58-bit)")
    print("  post-hoc 2-bit, no healing    : ~60-85% (degraded; our VQ post-hoc)")
    print("  bigger models lose LESS (more redundancy) -> retention rises with scale")
    print("  MoE: store TOTAL on disk, keep only ACTIVE experts resident -> RAM/speed set by ACTIVE")
    print("=" * 92)
    OUT.write_text(json.dumps({"assumptions": {"eff": EFF, "bits": BITS, "hardware": HW}, "models": rows},
                              indent=2, sort_keys=True), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
