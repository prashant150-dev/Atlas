"""Part-2 BEAST SPEED — step 2: the memory-bandwidth reality check.

A 400B model does NOT fit in 8GB RAM, so experts live on disk. The decisive question
for tok/s is: do we pay disk bandwidth PER TOKEN (reload experts every step) or ONCE
PER TASK (task-conditional routing keeps the active experts RAM-resident)?

Measured on this PC:
  * LUT kernel compute throughput (large matrices) : 3637 M active-params / sec
  * sequential disk read                           : ~1269 MB / s

We combine these honestly for both regimes and report achievable tok/s.

Run:  python projects/day18_speed/bandwidth_reality.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "bandwidth_results.json"

COMPUTE_PPS = 3637e6        # measured: active params/sec (LUT, large matrices, 4 cores)
DISK_BPS = 1269e6           # measured: bytes/sec sequential read
BITS = 2.0                  # bits/weight (mixed-precision VQ, Part-1)
BYTES_PER_PARAM = BITS / 8


def tok_s_compute(active):
    return COMPUTE_PPS / active


def tok_s_stream_serial(active):
    """experts reloaded from disk every token, load THEN compute (no overlap)."""
    load_s = active * BYTES_PER_PARAM / DISK_BPS
    comp_s = active / COMPUTE_PPS
    return 1.0 / (load_s + comp_s)


def tok_s_stream_overlap(active):
    """prefetch next token's experts while computing — bottleneck = slower stage."""
    load_s = active * BYTES_PER_PARAM / DISK_BPS
    comp_s = active / COMPUTE_PPS
    return 1.0 / max(load_s, comp_s)


def main():
    print(f"measured: compute {COMPUTE_PPS/1e6:.0f}M params/s | disk {DISK_BPS/1e6:.0f} MB/s "
          f"| {BITS} bits/weight\n", flush=True)
    print(f"{'active/token':>14} | {'RAM-resident':>12} | {'stream+overlap':>14} | "
          f"{'stream serial':>13}", flush=True)
    print(f"{'(per task)':>14} | {'(compute)':>12} | {'(prefetch)':>14} | {'(reload)':>13}",
          flush=True)
    rows = []
    for active in [80e6, 100e6, 200e6, 300e6, 500e6, 900e6]:
        r = {"active_M": active / 1e6,
             "ram_resident_tok_s": round(tok_s_compute(active), 1),
             "stream_overlap_tok_s": round(tok_s_stream_overlap(active), 1),
             "stream_serial_tok_s": round(tok_s_stream_serial(active), 1),
             "active_MB_at_2bit": round(active * BYTES_PER_PARAM / 1e6, 1)}
        rows.append(r)
        print(f"{active/1e6:11.0f} M | {r['ram_resident_tok_s']:10.1f}   | "
              f"{r['stream_overlap_tok_s']:12.1f}   | {r['stream_serial_tok_s']:11.1f}",
              flush=True)

    # the dream operating point
    print("\nHONEST READ:", flush=True)
    print("- Per-token RELOAD from disk (stream serial) is the slow wall — it adds the", flush=True)
    print("  load time of EVERY token. Avoid it.", flush=True)
    print("- Task-conditional routing (Day-15) loads a task's experts ONCE (they fit:", flush=True)
    print(f"  100M @2bit = {100e6*BYTES_PER_PARAM/1e6:.0f} MB, trivially in 8GB) and REUSES them for", flush=True)
    print("  all tokens -> per-token disk cost ~ 0 -> COMPUTE-BOUND.", flush=True)
    print("- So 40-50 tok/s needs ~80-100M active params/token, RAM-resident per task.", flush=True)
    print("  -> 80M: %.0f tok/s | 100M: %.0f tok/s (compute-bound)."
          % (tok_s_compute(80e6), tok_s_compute(100e6)), flush=True)

    payload = {"compute_params_per_sec_M": COMPUTE_PPS / 1e6, "disk_MB_per_sec": DISK_BPS / 1e6,
               "bits_per_weight": BITS, "rows": rows,
               "conclusion": "40-50 tok/s reachable at ~80-100M active params/token IF "
                             "task-conditional routing keeps the active experts RAM-resident "
                             "(load once per task, reuse all tokens). Per-token disk reload is "
                             "the wall to avoid."}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten {OUT}", flush=True)


if __name__ == "__main__":
    main()
