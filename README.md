# IntelMacAudioCpp

### 🚧 In progress

> [!CAUTION]
> **This is work in progress and is not ready for general use.**
>
> It is a research fork, validated on exactly one machine and one model. Kernels here have known
> unfixed defects, some paths are unmeasured, and a bad Metal kernel on a multi-GPU Intel Mac can
> hang the display driver and log you out. Read [Before you run anything](#before-you-run-anything).
> Do not use this as a base for anything you care about yet.

A fork of [audio.cpp](https://github.com/0xShug0/audio.cpp) that makes its vendored `ggml` Metal
backend work on **AMD GCN GPUs in Intel Macs** — wave64 hardware, where upstream assumes 32-wide
Apple simdgroups.

The target machine is a **2019 Mac Pro with a Radeon Pro Vega II**. The goal was not "does it run"
but "is it generating anything useful, and is it efficient".

## What this is for

Upstream ggml-metal hardcodes `#define N_SIMDWIDTH 32`, because every Apple GPU executes 32-wide
simdgroups. AMD GCN parts execute **64-wide wavefronts**. Every pipeline compiled on a Vega II
reports `th_width = 64`, so the mat-vec kernels were indexing 64-lane wavefronts with 32-lane
arithmetic.

The result was not "slow" or "slightly wrong". It was silent, total garbage:

| YuE2 output, 8 s clip | before | after |
|---|---|---|
| samples at full scale | **80.2%** | **0.0%** |
| distinct sample values | **38** | **49,464** |
| what it sounds like | a square wave | music |

Changing the model weights from q8_0 to q4_0 produced a **byte-identical** WAV, which is how we knew
the output could not depend on the model at all.

## Results

Measured on a Mac Pro (2019), Xeon W-3235, Radeon Pro Vega II 32 GB, running YuE2-3B q8_0 with an
f32 VAE at 8 NAR steps. Mean of three consecutive runs, spread 0.3%.

| stage | before | after | |
|---|---|---|---|
| **wall** | 26.86 s | **13.16 s** | **2.04×** |
| **RTF** | 3.36 | **1.64** | |
| NAR solve | 16.38 s | **4.98 s** | **3.29×** |
| VAE decode | 3.84 s | 2.33 s | 1.65× |
| semantic | 4.61 s | 4.61 s | — |
| AR decode | 2.75 s | 2.89 s | **0.95× — slower** |

**Why the NAR stage was slow is the interesting part**, and it was not bandwidth. Because
`has_simdgroup_mm` is false on AMD, upstream routes every matmul to a mat-vec kernel that processes
**one `src1` column per threadgroup**. At 200 frames that streams the entire weight matrix 200 times
per forward pass: 4823 GB of traffic for work that needs 24.1 GB. The card was already running at
294 GB/s, **45% of its measured 647 GB/s STREAM triad** — saturated, moving the wrong bytes.
`kernel_mul_mm_w64` removes that amplification.

### Against Apple silicon

Same request on both machines — same lyrics, style, seed, 200 semantic tokens, 8 NAR steps:

| stage | M4 Max | Vega II before | Vega II now | now vs M4 Max |
|---|---|---|---|---|
| **wall** | **7.67 s** | 26.86 s | **13.16 s** | **1.72× slower** |
| NAR solve | 1.18 s | 16.38 s | 4.98 s | **4.23×** |
| everything else | — | — | — | 1.3–1.8× |

The gap closed from 3.50× to 1.72× on short clips. The NAR solve is where it still loses, and that
is where the M4 Max's `simdgroup_matrix` instructions have no Vega equivalent. Our register-tiled
GEMM reaches 1.82 TFLOP/s against the 2.67 TFLOP/s a hand-written Mojo kernel reaches on this card,
so roughly 1.5× of that 4.2× is still recoverable; the rest is silicon.

### A full song, measured — 16 September 2026

> [!WARNING]
> **The short-clip figures above flatter this card.** On a real song it is **4.45x slower than the
> M4 Max**, not 1.72x. Here is the measurement.

`tonight-awake`, the audio.cpp benchmark song, q8_0 weights, f32 VAE, 8 NAR steps, `cot=full`:

| | Mac Pro 2019 / Vega II | Mac Studio / M4 Max |
|---|---|---|
| **song length** | **3:35.8** (215.8 s) | 3:36.8 |
| **compute time** | **13 min 34.7 s** (814.7 s) | **3 min 2.7 s** (182.7 s) |
| **RTF** | **3.77** | **0.84** |
| measured | 16 Sep 2026 | 15 Sep 2026 |

Stage breakdown, and this is the whole story:

| stage | Vega II | M4 Max | ratio |
|---|---|---|---|
| **semantic (AR)** | **542.1 s — 67% of wall** | ~95 s | **~5.7x** |
| &nbsp;&nbsp;of which ABC score | 110.2 s / 2,240 tok = **20.3 tok/s** | 96.4 tok/s | 4.7x |
| &nbsp;&nbsp;of which semantic tokens | 431.8 s / 5,396 tok = **12.5 tok/s** | 75.3 tok/s | 6.0x |
| NAR solve | 240.0 s | 63.4 s | 3.8x |
| VAE decode | 32.6 s | 22.4 s | 1.5x |

**Two thirds of a full song is spent in autoregressive decode, and that is the stage this work does
not help.** The GEMM fixed the NAR stage; attention is untouched. `FLASH_ATTN_EXT` is gated on
`simdgroup_matrix` support and aborts rather than falling back, so AMD must run with
`AUDIOCPP_DISABLE_FLASH_ATTN=1` and take the explicit attention path — whose cost grows with
context. At a 24,576-token context that is ruinous, and a 200-token clip never shows it. It also
explains why NAR came in at 240 s when the clip's 4.98 s / 200 frames would have predicted 134 s:
NAR attends over the same growing context.

**So the next piece of work on this card is decode attention, not more GEMM tuning** — and it is
designed for this machine rather than ported: see
[docs/vega-attention-design.md](docs/vega-attention-design.md) (the page with diagrams is
[docs/vega-attention-design.html](docs/vega-attention-design.html)). The control experiment behind
it: with flash attention on, the M4 Max's per-token cost is flat in context; with it off, it is
linear. The Vega II is refused flash attention by a vendor-family test and is linear too. The gap is
the software, not the silicon. (An earlier version of this paragraph said the Vega II was *faster*
than the M4 Max on the same algorithm; that is withdrawn — the M4 Max's FA-off path is a different
lowering and the two sweeps ran different batch sizes, so the slopes are not the same graph.)
At batch-1 decode attention is GEMV, so no `simdgroup_matrix` is involved. The design was revised
after verification: the KV cache is **F32** on Metal (224 KiB/token, not 112), the explicit path's
dominant cost is the `REPEAT`/`CONT` reshape chain (56% of step time), the runtime over-scans the
allocated cache, and the fixed floor that remains is 1,013 dispatches per token at 8.7 µs each.

**Measured, 16 Sep 2026, 20:11.** A standalone plain-Mojo benchmark of the design
(`attention_bench.mojo`, `/Volumes/S/oracles` branch `attention-bench`; results in
`RESULTS-vega2-attention.md`) decided the open question: **lane-per-dim with two keys per wave**
wins 24 of 25 cells, 1.82× best-vs-best at S = 8,192, reading K+V at 634 GB/s (76% of the blit
ceiling), verified against a Float64 CPU reference on every row. Attention at S = 8,192 costs
**~3 ms per token** — under 5% of the engine's context-dependent cost — so the engine's remaining
cost is the reshape/serial-scan lowering, not the arithmetic. Two findings the design lacked:
KV-head-fastest grid order is worth up to 3.9× (the unit of DRAM locality is the 4 KiB token row
across all 8 heads), and the template `enqueue_function` launch form costs 34–145 µs per dispatch
against 3–6 µs for a precompiled handle. Projected full song ≈ 404 s, RTF ≈ 1.87, with NAR and VAE
untouched. The kernel is not yet integrated into this tree.

The output is real audio at full length — peak 0.947, 0% clipped, 71,182 distinct levels, DC
3.5e-5 — so this is an honest speed number, not a fast wrong answer.

## What changed

Six pieces, and then a round of correctness fixes after an adversarial review found 15 defects in
them **before** any of it was trusted:

1. **Runtime SIMD-width probe** — compile a throwaway kernel, read `pipeline.threadExecutionWidth`,
   inject `N_SIMDWIDTH` at both `MTLCompileOptions` sites. Every pipeline's execution width is then
   asserted against the probe, so a device that picks its width per kernel fails loudly instead of
   computing garbage.
2. **Host-side width plumbing** — `GGML_METAL_NW(lib)` replaces the hardcoded 32 in threadgroup
   memory sizing and dispatch coverage, ~29 call sites.
3. **Wave-generic mat-vec kernels** — lane → block mapping for q8_0, the k-quants, f16/f32,
   `mul_mv_ext` and friends.
4. **`kernel_mul_mm_w64`** — a register-tiled GEMM for GPUs with no `simdgroup_matrix`: 256 threads
   (4 simdgroups × 64 lanes), 64×32 tile, per-operand staging types so an f32 operand is never
   narrowed to fp16 on the way into threadgroup memory.
5. **Device selection** — `MTLCreateSystemDefaultDevice()` returns the *display* GPU. On a Mac Pro
   that is often the wrong card. `GGML_METAL_DEVICE` picks by name or index.
6. **Concurrent dispatch** restricted to Apple-family GPUs.

Correctness fixes worth naming, because they were invisible:

- **`GGML_OP_SUM`** computed its simdgroup count as `(nth + 31) / 32` while `nth` came from the
  probed width, so at 64 it summed **16 uninitialised threadgroup floats** into every result. It
  measured `0 OK / 5 FAIL` before and `5 OK / 0 FAIL` after.
- **The `iq1`/`iq2`/`iq3` mat-vec kernels** seed shared values with a literal `32*sgitg + tiisg`. At
  64 lanes `iq2_xxs` indexes ~900 bytes past its threadgroup allocation. They are now refused in
  `supports_op` at non-32 widths and fall back to CPU. `iq4_nl` and `iq4_xs` are genuinely
  wave-generic and still work.
- **`GATED_DELTA_NET`** advertised shapes whose kernel is not instantiated, producing a null
  pipeline that was then dereferenced at encode time.

Apple silicon is unaffected **by the wave64 work**, and that is checked rather than asserted: the
shader compiled at `-D N_SIMDWIDTH=32` and disassembled gives **965 pre-existing functions, zero
differing** against the pre-fork baseline, with only the five new `kernel_mul_mm_w64_*` kernels
added — and those are unreachable there, since `has_mm_w64` requires `!has_simdgroup_mm` and a
probed width of 64.

**One later change is a deliberate exception and does touch the Apple path:** the q8_0 dequant
helper `dequantize_q8_0_t4` now reads its quants as two 16-bit loads instead of four 8-bit ones.
That helper is shared, so at `-D N_SIMDWIDTH=32` **14 functions differ** — the four
`kernel_mul_mv_ext_q8_0_f32_r1_*` and ten `kernel_flash_attn_ext_vec_q8_0_dk*_dv*`, the latter
reachable on Apple silicon even though they are dead here. The arithmetic is unchanged and the IR
says so: same `sext i8` + `air.convert.f.f32.s.i32`, same `fmul` operand order, no vectorised
multiply, no FMA contraction, no reassociation — only the loads and the byte extraction differ
(`load i8` 1 → 0, `load <2 x i8>` 0 → 2, and the 4-trip loop unrolled). Verified with a negative
control: perturbing one constant in an unrelated kernel reports exactly that one function.
**Not yet executed on Apple silicon** — one `FLASH_ATTN_EXT` run with `type_K/type_V=q8_0` on an
M-series machine is the missing evidence, and it is cheap there.

**A second exception of the same kind:** the 4x4 q8_0 helper `dequantize_q8_0` now reads its
sixteen quants as eight 16-bit loads instead of sixteen 8-bit ones. On the Vega II it stages the
weights for `kernel_mul_mm_w64_q8_0_f32`, the GEMM behind every q8_0 linear of the YuE2 NAR solve.
It is shared with Apple-reachable kernels, so at both `-D N_SIMDWIDTH=32` and `64` **38 functions
differ** and no others. They are `kernel_cpy_q8_0_f{32,16}`, `kernel_get_rows_q8_0`,
`kernel_mul_mm_q8_0_f{32,16}`, `kernel_mul_mm_id_q8_0_f{32,16}`, `kernel_mul_mm_w64_q8_0_f32`, and
the 30 `kernel_flash_attn_ext_impl` instantiations behind the 15 `kernel_flash_attn_ext_q8_0_dk*_dv*`
kernels. The same negative control holds. At each call site the IR goes `load i8` 1 → 0 and
`load <2 x i8>` 0 → 8. The `sext i8` + `air.convert.f.f32.s.i32` + scalar `fmul` (convert, d) is
unchanged, with no vectorised multiply and no FMA. Unrolled, the body crossed the frontend inline
threshold and was outlined at every call site, so the helper carries `always_inline` to keep the
inlined shape the loop had. A bare `inline` was not enough.

**Executed and measured on the Vega II.** Output is bit-identical to the previous build: a hash
probe of `kernel_mul_mm_w64_q8_0_f32` at the four NAR linear shapes with 5,418 rows, plus
`get_rows` and `cpy` q8_0, matches the previous build exactly, and a negative control
(`GGML_METAL_MM_W64_DISABLE=1`) changes every hash. `test-backend-ops` MUL_MAT is 955/955 on both
builds with identical per-test results, GET_ROWS q8_0 4/4, and the SUM and CPY controls are
unchanged. The q8_0 GEMM goes from 2.62 to 3.13 TFLOP/s (1.196x). At song-shaped length (5,400
frames, `cot=off`) the NAR stage goes from 194.6 s to 178.2 s (-8.5%), with the semantic stage and
VAE unchanged. Carried over to the full `cot=full` song, that projects NAR 230.7 s -> ~211 s.

**Still not executed on Apple silicon.** On Apple silicon the IR also changes shape, though not
arithmetic: in the 15 NSG=8 `kernel_flash_attn_ext_impl` instantiations the mask pointer now lives
in a stack slot. Nothing here shows whether Apple's backend re-promotes it. Before anyone relies on
it, run q8_0 `MUL_MAT`, `GET_ROWS` and `FLASH_ATTN_EXT` on an M-series machine.

## Before you run anything

> [!CAUTION]
> **Never run bare `test-backend-ops` on a multi-GPU Intel Mac.** Both GPUs share the
> `IOAcceleratorFamily2` driver. A compute hang on the headless card stalls the card driving your
> display, and the userspace watchdog kills WindowServer at 40 s — which logs you out and loses
> unsaved work. Use [`safe-sweep.sh`](safe-sweep.sh), which runs one op at a time and kills each at
> 30 s.

Other traps:

- **`GGML_METAL_SHARED_BUFFERS_ENABLE=1` is catastrophic here** — 59–95× slower. It moves 4.26 GB of
  weights to host RAM over PCIe. Never set it on a discrete card.
- **Exit code 0 proves nothing.** A run can exit clean, write a plausible file size and the right
  duration, and still be a full-scale square wave. Check RMS, full-scale count and level count.
- Flash attention is **half ported**. The decode-shaped path now has wave64 kernels
  (`kernel_flash_attn_ext_vec_w64` / `_vec_reduce_w64`) and `ggml_metal_device_supports_op()`
  admits exactly those nodes. The prefill-shaped path still lowers to the `simdgroup_matrix`
  kernel, has no wave64 kernel, and still aborts rather than falling back. So an AMD run needs
  `AUDIOCPP_DISABLE_FLASH_ATTN=1` to keep prefill and the NAR explicit, plus
  `AUDIOCPP_FLASH_ATTN_DECODE=1` to let AR decode take the flash branch. **Unmeasured on hardware
  at the time of writing** - verify with `test-backend-ops` on `FLASH_ATTN_EXT` first.

## Usage

```bash
GGML_METAL_DEVICE=Vega AUDIOCPP_DISABLE_FLASH_ATTN=1 \
  ./audiocpp_cli --task gen --family yue2 --model models/Yue2-3B-GGUF \
  --backend metal --lyrics "..." --out out.wav --metrics
```

Confirm the port is actually armed — all four lines must appear:

```
selected device: AMD Radeon Pro Vega II
simdgroup matrix mul. = false
simd group width      = 64
wave64 mat-mul        = true
wave64 flash attn     = true
```

### Dials

| variable | what it does |
|---|---|
| `GGML_METAL_DEVICE` | select GPU by name substring or index |
| `AUDIOCPP_DISABLE_FLASH_ATTN` | presence test; disables every flash branch. Still required on AMD to keep prefill and the NAR off the unported `simdgroup_matrix` kernel |
| `AUDIOCPP_FLASH_ATTN_DECODE` | value-aware; overrides the above for decode-shaped attention. Set to `1` on AMD to take the wave64 flash path |
| `AUDIOCPP_FLASH_ATTN_PREFILL` | value-aware; same for prefill-shaped attention. Leave unset on AMD |
| `AUDIOCPP_FLASH_ATTN_NAR` | value-aware; same for the NAR stage. Leave unset on AMD |
| `GGML_METAL_FA_W64_DISABLE` | value-aware; turns the wave64 flash kernels off at the ggml layer, so `supports_op` goes back to refusing. The A/B dial for the flash path |
| `GGML_METAL_MM_W64_DISABLE` | fall back to mat-vec. **The** A/B dial: width stays 64, only GEMM selection changes |
| `GGML_METAL_MM_MIN` | mat-vec → mat-mul crossover. Still upstream's 8, **unmeasured** on this card |
| `GGML_METAL_CONCURRENCY_ENABLE` | re-enable concurrent dispatch on non-Apple GPUs |
| `GGML_METAL_N_CB` | command buffers per graph |

## Known unfixed

- **`CPY` and `CONT` fail with sentinel mismatches** — out-of-bounds *writes* — on permuted
  f32/f16/bf16 copies and on `i32 CONT` with `use_view_slice=1`. These fail **identically before and
  after** every change here, so they are pre-existing, not introduced. They are still real, and more
  suspect now: `CONCAT` had a bug of exactly this class (below), and it corrupted real output.
- **AR decode is 5% slower** than before the port, outside the 0.4% noise floor. The concurrency
  gate is the suspect.
- **`mm_min` is unmeasured.** Upstream's 8 was tuned for Apple tile shapes, not a 64×32 tile.
- ~~**Runs are not bit-reproducible.**~~ **Fixed, and the earlier explanation was wrong.** Fixed-seed
  runs used to differ at SNR ~62 dB. That was attributed here to reduction-order floating point. The
  real cause was `kernel_concat`: it was float-only while `supports_op` admitted every type, so the NAR's
  f16 K/V concatenation copied 4 bytes at a 2-byte stride and raced at the seam between its two
  sources, 896 times per song. With a typed `kernel_concat`, same-seed runs are **byte-identical** (3/3 at
  1,600 frames, 2/2 at 5,400).
- Only YuE2 has been exercised. Only one GPU. Only macOS.

## Licence and attribution

This fork inherits its licences unchanged, and adds none:

- **audio.cpp** — Apache License 2.0, Copyright 2026 ShugoAI LLC. See [LICENSE](LICENSE).
- **ggml** (vendored at `external/ggml`) — MIT, Copyright (c) 2023-2026 The ggml authors. See
  [external/ggml/LICENSE](external/ggml/LICENSE).

Modified files are listed in [NOTICE](NOTICE), as Apache 2.0 section 4(b) requires. The upstream
project's own README is preserved as [README-upstream.md](README-upstream.md).

All credit for audio.cpp and ggml belongs to their authors. This fork only makes their Metal backend
count to 64.
