# Attention for the Vega II — designed from requirements, not ported

*16 Sep 2026. Revised the same evening after three requirements agents checked the machine, the
workload and Apple's assumptions against the user's oracles and the engine source. What they changed
is listed first, because some of it reverses the original.*

## Outcome — the bench decided it, 16 Sep 2026, 20:11

`attention_bench.mojo` (plain Mojo: `std.gpu`, `max.gpu.host`, `max.gpu.sync`, no ObjC, no Metal API;
784 lines; `/Volumes/S/oracles` branch `attention-bench`, commits `2de3831` + `ff4291c`; results in
`RESULTS-vega2-attention.md`) implemented both Phase A variants on the native `[128, 8, S, B]` layout
with BLOCK, dtype, batch and `valid_steps` as parameters, verified every timed row against a Float64
CPU reference (maxerr ≤ 2.7e-8, NMSE ≤ 4e-14, zero wrong in 124+ rows including ragged `valid`,
partial blocks, B=2, fp16 and scan > valid), and measured. Every run under `timeout 30`.

**Phase A(b) — lane ↔ dim, two keys per wave — wins.** 24 of 25 cells; best-vs-best at S = 8,192
**105.8 µs vs 192.8 µs per layer = 1.82×** (2.0–2.2× at the same BLOCK). At S = 8,192 it reads K+V
at **634 GB/s** — ~91% of the 699 GB/s the same kernel reaches at B = 2, 76% of the 830 GB/s blit
ceiling. KEY tops out at 348 GB/s and gets worse with larger blocks: exactly the L1 failure the
design predicted for it.

**Per token (×28 layers), fp32, B = 1: 0.56 / 0.80 / 1.08 / 1.73 / 2.96 ms at S = 256 … 8,192.**
Against the engine's ~62–66 ms/token context-dependent cost at S ≈ 8,000, the attention itself is
**~3 ms — under 5% of it**. The context cost the engine pays is the reshape chain and the
serial-scan lowering around the arithmetic, not the arithmetic or the bytes.

**Three things the bench found that the design did not contain:**

1. **Grid order was the larger effect.** Launching key-block-fastest put the 8 KV heads of one
   token range `nblocks` apart, so resident threadgroups each took a 512 B slice out of every 4 KiB
   token row. KV-head-fastest makes the resident set consume whole rows: **3.9× at BLOCK 64,
   S = 8,192** (502.6 → 128.0 µs). The unit of DRAM locality in the native layout is the token row
   across all 8 heads. The design's "1 KiB contiguous per wave instruction" was really two 512 B
   chunks 4 KiB apart; what matters is keeping the 8 heads of a token range co-resident.
2. **The matmul occupancy ladder does not transfer.** At S = 8,192 the curve from 16 TG/CU down
   to 1 is flat-to-inverted (129.6 → 109.8 µs) and B = 2 — twice the threadgroups, same per-TG
   work — buys only 10–18%. With 4 float4 loads in flight per lane the memory pipe saturates by ILP
   at ~2 TG/CU. R2's "≥ 256 threadgroups, ideally 1,024" does not govern this kernel above
   S ≈ 4,000. BLOCK 64 for S ≤ 1,024; above that 128–512 are within 3% and BLOCK 256 is a fair
   single choice.
3. **The launch form.** `ctx.enqueue_function[kernel](...)` re-resolves the function per call at
   34–145 µs per dispatch; `compile_function` once plus `enqueue_function(handle, ...)` is 3–6 µs.
   The control copy kernel moved from 573–591 to 634–660 GB/s from that change alone. This bears
   directly on the engine's 8.7 µs × 1,013 dispatch floor.

Also measured: **fp16 K/V** is 1.37× faster for half the bytes (435 GB/s) — issue-bound, not
bandwidth-bound, so fp16 buys less than 2× here; **B = 2** reaches 696–699 GB/s, above the 647
triad, so B = 1 is parallelism-limited at 128 threadgroups rather than bandwidth-limited;
**over-scan** costs its full length (scanning 8,192 for 4,096 valid: 1.7×), confirming R5a;
`exp2` lowers and is correct on this card (first run ever, worst rel 5.3e-6).

**Projection, honestly bounded.** Semantic stage 5,396 tokens at 11.24 + ~2.1 ms ≈ 100 s (from
431.8); ABC ≈ 31 s (from 110.2); NAR 240 s and VAE 32.6 s untouched (prefill-shaped, kept on the
explicit path). **Full song ≈ 404 s, RTF ≈ 1.87** from 3.77. The floor after this is the
11.24 ms/token of dispatch and weights — the next problem, and the launch-form finding is the
first lead on it.

**Before the engine uses this kernel** (from review): masked keys currently load a clamped row and
`0 × NaN` would poison the output if the over-allocation holds NaN bits — clamp to `valid − 1` or
select V to 0 when masked. `valid == 0` and `scan < valid` are caller contracts.

## Corrections applied after verification

1. **The KV cache is F32 on Metal, not F16.** 512-byte rows, 224 KiB per token, 8 KiB per layer per
   token. The F16 storage path exists in the engine but is gated to CUDA/HIP/Vulkan. The kernel takes
   the dtype as a parameter; F16 is free 2x headroom later (R5 below).
2. **The explicit path's dominant cost is data movement the design had not identified.** Per layer
   the eager path `REPEAT`s K and V from 8 heads to 16 and does four `CONT`s: six dispatches that
   exist only to reshape, measured at **56% of step time**. QK^T is not starved (104,512 independent
   dot products at batch 2); only the AV reduction is. New R0: consume the native layout
   `ne=[128, 8, S, B]`, `nb=[4, 512, 4096, 4096·S]` and index GQA; never materialise.
3. **The runtime over-scans.** Cost tracks *allocated* cache steps (prefix + 5,120), not live context.
   A `valid_steps` bound in the kernel recovers ~24% of the context term. New R6.
4. **The operating range is S = 4,000–12,000, common case 7,837** — not 24,576, which is the NAR
   ceiling the AR never reaches.
5. **The fixed floor is dispatch.** After a perfect attention kernel, 11.24 ms/token remains: 2.43 ms
   of weight traffic (the AR uses a compact 32,769-row `lm_head` view, 1.57 GB) and **8.81 ms of
   1,013 serialised dispatches at 8.7 µs each**. Attention fusion removes 168 of them.
6. **The "Vega II beats the M4 Max on the same algorithm" claim is withdrawn.** The M4 Max's FA-off
   path is a different lowering (`FlashGroupedViewKV`, no CONT/REPEAT chain), and the Vega sweep ran
   CFG batch 2 where the full song ran batch 1. The slopes were not the same graph. The conclusion —
   the gap is the software, not the silicon — stands on the FA-on/FA-off control alone.
7. **Phase A's mapping cannot be chosen on paper.** The lane-per-key variant relies on L1 holding
   4 KiB per wave; at 8 threadgroups per CU that is 128 KiB against a 16 KiB L1. And at head_dim 128
   a 64-wide wave handles **two KV rows per instruction** in the lane-per-dim mapping — 1 KiB
   contiguous — so wave64 is a structural fit for that variant, not a cost. The register file is
   unobservable on this toolchain (Apple's AIR→GCN is a black box; the only proxy is the pipeline's
   reported max threads), so the mapping is a **measured** decision. Both variants are comptime
   parameters of `attention_bench.mojo`.
8. Measured on this card and now design inputs: the ILP knee is **8 independent chains per lane**
   (94% of peak; 64 chains buys 6% more at 8x the registers); the occupancy ladder for 256-thread
   threadgroups is **1 TG/CU → 0.97x, 4 → 1.70x, 16 → 1.95x**, so the grid must reach ≥256
   threadgroups and ideally ~1,024; the block size is therefore a swept parameter, not structural.
9. The bandwidth ceiling is instrument-dependent by 1.28x (647 GB/s STREAM triad vs 830 GB/s copy).
   This document quotes **647** and says so.
10. Hazards measured on this path that the kernel must satisfy: threadgroup size derived from the
    probed wave width (a constant silently halves at 64 → NaNs, no error); cross-lane ops and barriers
    `convergent` (else barriers are cloned per branch and lanes desynchronise silently);
    `simd_ballot` in its i64 form; `llvm.vector.reduce.*` expanded by hand.


The rule for this document: every design decision must trace to a requirement, and every requirement
must trace to a measurement or to the machine. Nothing is here because Apple's kernel does it.

---

## 1. The requirements

### R0 — Consume the cache in its native layout; never materialise a 16-head copy

Measured: six of the nine per-layer attention dispatches in the explicit path are `REPEAT` (8→16 heads)
and `CONT`, and they are 56% of step time. The cache is `[128, 8, S, B]` with the eight KV heads of a
token contiguous in 4,096 bytes. Query head *h* reads KV head *h/2* by indexing. One dispatch per layer
for the whole attention region replaces nine.

### R1 — Parallelism must come from the context dimension, because nothing else has any

Measured: at batch 1 there are 16 heads of attention work per layer, 32 under CFG. The card has
64 CUs ⟨verify⟩, 4 SIMDs each, up to 10 waves per SIMD ⟨verify⟩ — about **2,560 wave slots**. Sixteen
threadgroups is ~1% of that, and each one then walks the whole KV context serially. That is the
entire cause of the 80 ms/token: latency-exposed, not bandwidth-limited (69× off the traffic floor).

At batch 1 the query is one vector. Heads are 16. Batch is 1–2. **The only dimension with parallelism
in it is the context**, which is 1,000–8,000 long during a song. So the kernel must split the KV
range into blocks and give each block its own workers, then combine.

Quantified: at context 8,000 and 16 heads, blocks of 64 keys give 125 × 16 = 2,000 work items.
Blocks of 256 give 500. Context 200 gives only 48 either way — acceptable, because short context is
cheap regardless.

### R2 — Hide DRAM latency by wave switching, which means a small register footprint per wave

GCN hides memory latency by having other waves ready, not by cache. Occupancy is set by VGPRs per
lane: ⟨verify⟩ roughly ≤24 VGPRs → 10 waves/SIMD, 32 → 8, 64 → 4, 128 → 2. The design must keep
each lane's live state small enough for ≥4 waves per SIMD, preferably 8.

This is the tension the design has to resolve: **register blocking** (the lever the Mojo GEMM used
to reach 2.67 TFLOP/s — 4×2 accumulators, few waves) versus **occupancy** (many light waves). The
GEMM is compute-shaped and wants blocking. Attention at batch 1 is *traffic-shaped* — 619 MB of KV
per token against 7 MFLOP of arithmetic — so it wants occupancy. Different problem, different answer.

### R3 — Every K and V row must be read coalesced, exactly once

A K or V row is head_dim 128 × **f32 = 512 bytes** as the engine allocates it on Metal today (f16 = 256 B is a parameter, not the default). A wave64 memory instruction is
64 lanes; the coalescing unit is ⟨verify⟩ 64-byte segments. The lane→data mapping decides whether
a row is one clean 256 B access or 64 scattered ones. The mapping is therefore a *requirement*, not
a detail.

### R4 — GQA 2:1 means every KV row serves two query heads; read it once

`num_key_value_heads = 8`, `num_attention_heads = 16`. Two query heads share each K/V head. If they
are processed by the same threadgroup, K and V traffic halves: 619 MB → 310 MB per token at 8,000
context. Nothing in the algorithm prevents it; it is a scheduling choice.

### R5a — The kernel takes `valid_steps` and stops there

Measured: the runtime allocates prefix + 5,120 cache steps and the explicit path scans all of them from
the first generated token, so cost tracks allocation, not live context. A per-block early exit on
`valid_steps` is worth ~24% of the context term on the full song's semantic stage — the largest single
non-bandwidth change available.

### R5 — Partial results from context blocks must combine exactly

Splitting context means each block produces a partial (running max *m*, running sum *l*, unnormalised
output *o*[128]) and these must merge with the standard rescaling:
*m = max(m₁,m₂); l = l₁e^(m₁−m) + l₂e^(m₂−m); o = o₁e^(m₁−m) + o₂e^(m₂−m)*. This is the online-softmax
identity — intrinsic to attention, owes nothing to any vendor.

Measured: host dispatch cost is not a factor here (`n_cb` flat, CPU 11% busy), so a **second reduce
kernel is acceptable**. We do not need to contort the design to fuse everything into one dispatch.

### R6 — Accumulate in fp32; exp must not become the bottleneck

Softmax needs one `exp` per key per head: 8,000 × 16 × 28 layers ≈ 3.6 M per token. GCN's
transcendental unit is ⟨verify⟩ quarter-rate. Even so that is microseconds. Accumulators and the
running max/sum stay fp32; K/V are read as stored.

### R7 — Correctness oracle exists already, and so does the target curve

The explicit-attention path produces correct audio, so it is the per-tensor reference. And the M4
Max FA-on measurement is the **target shape**: ms/token flat in context. Success is measured as
"the Vega II's curve goes flat", not as a speedup number picked in advance.

---

## 2. What the card offers, and which requirement each feature answers

| Feature of the Vega II | Answers | How |
|---|---|---|
| 64 CUs × 4 SIMDs × 10 waves ⟨verify⟩ | R1 | Enough slots for 2,000+ light work items — one per (head-pair, context block) |
| **256 KB register file per CU** ⟨verify⟩ — large by any standard | R2 | Even at 8 waves/SIMD each lane still has 32 VGPRs; the state we need is ~16. Occupancy and headroom both |
| **64 KB LDS per CU**, conflict-free broadcast reads | R3, R4, R5 | Holds Q for both heads of a pair (1 KB), the per-block probabilities (256 B), and the intra-threadgroup partials |
| 64-wide waves | R3, R5 | A wave reads a full 256 B row in one instruction at 4 B per lane; a block of 64 keys is one wave with one key per lane — no tail logic inside the block |
| 647 GB/s measured triad, 4 MB L2 ⟨verify⟩ | R3 | With 2,000 waves in flight the KV read becomes bandwidth-shaped instead of latency-shaped; at that point 310 MB is ~0.5 ms |
| Quarter-rate transcendentals ⟨verify⟩ | R6 | 3.6 M exps per token is ~2 µs at that rate. No exp2/polynomial tricks needed unless measured otherwise |
| `simd_shuffle` / `simd_max` / `simd_sum` proven correct at 64 lanes today | R5 | The reductions this design needs are the ones `GGML_OP_SUM` and the mat-vec kernels now pass with |

---

## 3. The design

### 3.1 Decomposition

```
grid:        (KV-head, context-block)      = 8 × ⌈context/256⌉        → 250 threadgroups at 8,000
threadgroup: 256 threads = 4 waves          (same shape as kernel_mul_mm_w64, already validated)
wave:        64 keys, one key per lane      (block of 256 keys = 4 waves × 64)
each threadgroup serves BOTH query heads of its KV head   (R4)
```

Block size is a **swept parameter** (64–512 keys per threadgroup), not a structural constant: the
measured occupancy ladder on this card for 256-thread threadgroups is 1 TG/CU → 0.97x, 4 → 1.70x,
16 → 1.95x, so the grid must reach ≥ 256 threadgroups at the common S ≈ 7,800 and ideally ~1,024.
At S = 8,000 and 8 KV heads: BLOCK 256 gives 256 threadgroups; BLOCK 64 gives 1,000. The bench
measures which pays.

### 3.2 Three phases — Phase A's mapping was decided by measurement: lane ↔ dim (see Outcome)

Apple's kernel uses one lane→data mapping throughout because `simdgroup_matrix` imposes it. We have no
such constraint. Two candidate mappings for the score phase, both implemented as comptime parameters
of `attention_bench.mojo`:

**Phase A(a) — lane ↔ key.** Lane *j* owns key *j* of its wave's 64, streams its own 512 B row, forms two
dot products against Q₀, Q₁ read from LDS as broadcasts. 256 FMAs per lane, zero cross-lane ops. Risk:
64 rows 4,096 B apart per instruction, relying on L1 to hold 4 KiB per wave — 128 KiB at the
occupancy R2 demands, against a 16 KiB L1. **Probably loses under load.**

**Phase A(b) — lane ↔ dim, two keys per wave.** At head_dim 128, 32 lanes × float4 cover one row; a
64-lane wave covers **two keys per instruction**, 1 KiB contiguous. Per key pair: 4 FMAs per lane then a
5-step shuffle reduction over 32 lanes. Keep ≤ 8 independent accumulator chains per lane — the measured
ILP knee. Wave64 halves the softmax and barrier count per key relative to Apple's 32-wide C. **The
structurally coalesced one.**

**Phase B — softmax.** Wave max, exp, wave sum — once per 64 keys, ladders generated from the wave
width. p to LDS (512 B per head).

**Phase C — output.** The mapping that matches the chosen Phase A: lane-per-dim with V rows read as
whole coalesced rows across the wave, *p<sub>j</sub>* as an LDS broadcast. Zero cross-lane ops.

Per-lane live state is the budget: the register file cannot be read on this toolchain, so the bench
brackets the blocking factor by throughput, exactly as the fork's `nr0` sweep did (8 optimal, 16 and 32
worse, with no register visibility at all).

### 3.3 Combining

Inside the threadgroup: the 4 waves each hold a partial (m, l, o[128]) per head. Merge them through
LDS with the R5 identity → one partial per threadgroup per head. Write it to a scratch buffer
`[16 heads × ⌈context/256⌉ × 130 floats]` — ~1 MB at 8,000 context.

**Reduce kernel:** one threadgroup per head, 64 lanes, each lane merges a strided subset of the
partials with the R5 identity, wave-reduce, normalise by *l*, write the head's 128 outputs. Trivially
small. Measured dispatch cost says a second kernel is free; taking it keeps the main kernel simple.

### 3.4 Tail and mask

A block of 256 at the end of a context that is not a multiple of 256: lanes past the end contribute
s = −∞ (so p = 0) and skip their loads. The causal mask at decode is "all keys ≤ current position",
which is the entire cache — so at batch 1 the mask is simply the tail guard. The explicit path's
mask tensor is not needed; ggml's `FLASH_ATTN_EXT` op signature carries one, which the kernel can
honour if present and ignore at decode.

### 3.5 What stays on the explicit path

Prefill (many queries at once) has real matrices in it and the explicit path handles it correctly
today. It is not in the 67%. It stays as it is. The host gate becomes: use this kernel when
`n_queries ≤ 2` (decode, including CFG) and the device is wave64; otherwise the existing path.

---

## 4. What is genuinely harder here, stated honestly

- **Phase A's stride (variant a only).** Lane↔key means 64 lanes reading rows 4,096 B apart (the head stride in the native layout). It relies on L1 holding
  64 lanes × 64 B = 4 KB of segments across the inner loop ⟨verify L1 is ≥16 KB and the access
  pattern does not thrash it⟩. If it does thrash, the fallback is staging K through LDS with a
  coalesced lane↔dim load and a transpose — 16 KB per block, well within 64 KB — at the cost of a
  barrier. That is the one place the design has a plan B.
- **The reduction count is 6 steps not 5** per wave-max / wave-sum — but once per 64 keys rather than 32, so 3.3x cheaper per key than Apple's. Two per block per head. Not a
  concern at this ratio of FMAs to shuffles, but it is the concrete cost of wave64.
- **Two lane mappings means two mental models in one kernel.** It is more to get right than one
  mapping. It is also the source of most of the win.

## 5. How we will know

1. Op-level: `test-backend-ops` on `FLASH_ATTN_EXT` at the real shapes, one op under the 30 s
   watchdog, against the CPU reference. NMSE, not "looks fine".
2. Per-tensor: the new kernel against the explicit path, same inputs, on the card.
3. **The curve**: ms/token at 200 / 800 / 1,600 / 3,200 / 6,400. It must go flat. That is the only
   speed claim this design makes.
4. Then the waveform check, then the song, then the headphones — in that order, as always.
