"""Day-5 R8: HONEST projections for a 400B model (NOT benchmarks).

We cannot run a 400B model on an 8 GB CPU, so nothing here is measured on a 400B
model. These are extrapolations from our measured constants (D1 entropy floor,
GPT-2 tok/s, packed-ternary bits/weight) plus this machine's physical limits
(RAM, bandwidth, disk). Every number is an ESTIMATE and labelled as such.

Run: python projects/day5_reasoner_memory/r8_projection_400b.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "r8_projection.json"

# ---- this machine (measured / spec) ----
RAM_GB = 8.0
DISK_FREE_GB = 56.0
BW_GBs = 18.0            # realistic DDR3 dual-channel read bandwidth
DISK_BW_GBs = 0.5       # SATA SSD streaming
ENTROPY_FLOOR_BITS = 2.04   # D1: irreducible bits/weight for GPT-2-like weights

P = 400e9               # 400 billion parameters


def bytes_per_weight(bits):
    return bits / 8.0


def storage_gb(n_params, bits):
    return n_params * bytes_per_weight(bits) / 1e9


def main():
    rows = []

    # ---- A. monolithic DENSE 400B: storage at various precisions ----
    storage = {f"{b}bit": round(storage_gb(P, b), 1) for b in (16, 8, 4)}
    storage["ternary~2bit"] = round(storage_gb(P, 2), 1)
    storage["entropy_floor_2.04bit"] = round(storage_gb(P, ENTROPY_FLOOR_BITS), 1)

    # bits/weight you'd need to fit 400B into 8 GB RAM:
    need_bits_ram = RAM_GB * 8e9 / P
    need_bits_disk = DISK_FREE_GB * 8e9 / P

    # dense speed: every token reads ALL weights once
    def dense_tok_s(bits, bw):
        gb_per_token = storage_gb(P, bits)
        return bw / gb_per_token            # tokens/sec = bandwidth / bytes-per-token
    dense_speed = {
        "ternary_from_RAM_if_it_fit": round(dense_tok_s(2, BW_GBs), 4),
        "ternary_from_DISK": round(dense_tok_s(2, DISK_BW_GBs), 4),
        "int8_from_DISK": round(dense_tok_s(8, DISK_BW_GBs), 4),
    }

    # ---- B. MoE-sparse 400B: only a fraction active per token ----
    moe = []
    for active_frac in (0.05, 0.02, 0.01):
        active = P * active_frac
        active_gb_tern = storage_gb(active, 2)
        # speed if active experts are resident in RAM (needs R7 kernel + enough RAM)
        tok_s_ram = BW_GBs / active_gb_tern
        moe.append({
            "active_fraction": active_frac,
            "active_params": f"{active/1e9:.0f}B",
            "active_ternary_GB_per_token": round(active_gb_tern, 2),
            "est_tok_s_if_active_in_RAM": round(tok_s_ram, 1),
            "active_fits_8GB_RAM": active_gb_tern <= RAM_GB,
        })

    # ---- C. our architecture: small native reasoner + external memory ----
    reasoner = []
    for rb in (3e9, 7e9):
        gb = storage_gb(rb, 2)
        reasoner.append({
            "reasoner_params": f"{rb/1e9:.0f}B",
            "ternary_size_GB": round(gb, 2),
            "fits_8GB_RAM": gb <= RAM_GB,
            "note": "knowledge lives in external memory (disk), not these weights",
        })

    payload = {
        "DISCLAIMER": "PROJECTIONS / ESTIMATES ONLY — no 400B model was run; 8GB CPU cannot run one.",
        "machine": {"RAM_GB": RAM_GB, "disk_free_GB": DISK_FREE_GB,
                    "ram_bandwidth_GBs": BW_GBs, "disk_bandwidth_GBs": DISK_BW_GBs},
        "A_monolithic_dense_400B": {
            "storage_GB_by_precision": storage,
            "bits_per_weight_to_fit_8GB_RAM": round(need_bits_ram, 3),
            "bits_per_weight_to_fit_56GB_disk": round(need_bits_disk, 3),
            "entropy_floor_bits": ENTROPY_FLOOR_BITS,
            "verdict": "IMPOSSIBLE on this PC: needed %.3f bits/wt << proven floor %.2f"
                       % (need_bits_ram, ENTROPY_FLOOR_BITS),
            "dense_speed_tok_s": dense_speed,
        },
        "B_moe_sparse_400B": moe,
        "C_reasoner_plus_memory": reasoner,
        "target_tok_s": "40-50 (the dream)",
    }
    print(json.dumps(payload, indent=2))
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
