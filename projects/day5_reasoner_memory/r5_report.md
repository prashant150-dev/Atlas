# Day 5 — R5: The Grand Merge (compressed reasoner in the full pipeline)

We brought the compression track (int8 / packed-ternary, `gpt2_packed`) into the
R4 reasoner+memory pipeline and re-flew the jet with three reasoner variants.
Retrieval was held fixed (fp32 embeddings + learned head + hybrid); the only
variable is the reasoner that reads the retrieved fact and answers.

## Compression of the reasoner (disk)

| variant | disk size | ratio | relative L2 error |
|---|---|---|---|
| int8 | 128.4 MB | 5.1× | **0.077** |
| ternary | 36.1 MB | 18.0× | **0.481** |

## End-to-end, full pipeline (240-fact memory, mixed named+alias workload)

| reasoner | disk | runtime RAM | tok/sec | answer accuracy |
|---|---|---|---|---|
| fp32 | 501 MB | 498 MB | 17.1 | **0.912** |
| **int8** | **128 MB** | 498 MB | 17.6 | **0.912** |
| ternary | **36 MB** | 498 MB | 17.5 | **0.000** |

## The honest result

- **int8 is a free win: 4× smaller on disk, accuracy IDENTICAL (0.912), same
  speed.** The reasoner's copy-from-context job tolerates int8 (rel err 0.077)
  perfectly. This is the safe operating point.
- **Ternary is 18× on disk (36 MB!) but the reasoner COLLAPSES to 0.000.** Naive
  post-hoc ternary destroys GPT-2 (rel err 0.48) — it can no longer read the
  retrieved fact at all. Exactly the D1 / P1.1 wall: ~2 bits post-hoc = signal
  gone. A ternary reasoner would have to be *healed* (D2) or *natively trained*
  (D3) to be usable — and P1.1 showed even healed ternary GPT-2 tops out ~30%.
- **RAM and speed are identical across all three (498 MB, ~17 tok/sec)** because
  `load_packed_gpt2_model` dequantizes back to fp32 — this CPU has no native
  low-bit kernel. **On this machine the compression win is DISK only, not RAM or
  speed.** (RSS grew across variants only because the single process held all
  three models at once; per-variant fp32 weight RAM is the same 498 MB.)

## What the merged jet says about the dream

- **Storage of everything is solved/cheap:** knowledge ~55 bits/fact (1B ≈ 7 GB)
  *and* the reasoner int8 at 128 MB — the whole system is disk-light.
- **The two missing dream levers are now precisely named:**
  1. **A usable low-bit reasoner** — post-hoc ternary is dead (0.000); the route
     is *native* low-bit training (D3 AetherNet beat post-hoc) or strong healing,
     not naive quantization.
  2. **Low-bit inference kernels** — without them, ternary buys no RAM/speed on
     CPU. The dream's 40–50 tok/sec and fit-in-RAM need real packed-ternary
     matmul kernels (bitnet.cpp-style). This is the honest hard frontier.

## The picture after R5

```
knowledge  -> external memory      55 bits/fact, 1B~7GB        ✅ cheap
retrieval  -> hybrid (lex+learned) 0.85, alias-robust          ✅ works
reasoner   -> int8                 128 MB, 0.912 acc, free      ✅ safe compression
           -> ternary              36 MB but 0.000 acc          ❌ needs native/healed + kernels
speed/RAM  -> 17 tok/s, 498 MB     compression doesn't help on CPU (no kernel)  ⚠️ frontier
```

## Next
- **R6:** a *natively low-bit* reasoner (AetherNet-style, D3) inside the pipeline
  instead of post-hoc ternary — does native survive where post-hoc died (0.000)?
- **R7:** real packed-ternary matmul kernel (the RAM+speed lever) — even a slow
  reference kernel proves the path.
