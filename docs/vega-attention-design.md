# Attention for the Vega II — designed from requirements, not ported

*16 Sep 2026. Numbers marked ⟨verify⟩ are being confirmed against the user's own oracles.*

The rule for this document: every design decision must trace to a requirement, and every requirement
must trace to a measurement or to the machine. Nothing is here because Apple's kernel does it.

---

## 1. The requirements

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

A K or V row is head_dim 128 × f16 = 256 bytes ⟨verify dtype⟩. A wave64 memory instruction is
64 lanes; the coalescing unit is ⟨verify⟩ 64-byte segments. The lane→data mapping decides whether
a row is one clean 256 B access or 64 scattered ones. The mapping is therefore a *requirement*, not
a detail.

### R4 — GQA 2:1 means every KV row serves two query heads; read it once

`num_key_value_heads = 8`, `num_attention_heads = 16`. Two query heads share each K/V head. If they
are processed by the same threadgroup, K and V traffic halves: 619 MB → 310 MB per token at 8,000
context. Nothing in the algorithm prevents it; it is a scheduling choice.

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

At context 8,000: 250 threadgroups × 4 waves = **1,000 waves**, each lane at ~16 VGPRs, so up to
8 waves/SIMD are schedulable → the machine is full ⟨verify against the occupancy table⟩. At 1,000
context it is 32 threadgroups / 128 waves — under-filled, but that regime costs 15 ms/token today
and is not where the time goes.

### 3.2 Three phases, two lane mappings — use each where it is natural

This is the part a straight port gets wrong. Apple's kernel uses one lane→data mapping throughout
because `simdgroup_matrix` imposes it. We have no such constraint, so each phase gets the mapping
that makes *its* memory access coalesced and *its* arithmetic shuffle-free.

**Phase A — scores, lane ↔ key.** Lane *j* owns key *j* of its wave's 64. It streams its K row
(256 B, contiguous per lane; across lanes the 64 rows are 256 B apart, and each lane consumes its
whole 64 B segments over the loop so L1 absorbs the stride ⟨verify L1 size⟩) and forms two dot
products against Q₀ and Q₁, which sit in LDS and are read as conflict-free broadcasts. **128 FMAs
per lane per head, zero cross-lane operations.** Every lane is independent. This is the GEMV
mapping an AMD programmer writes first.

**Phase B — softmax, two reductions per block.** Each lane holds s₀ⱼ, s₁ⱼ. Wave-max via `simd_max`
(6 steps at 64 lanes — once per block, not per key), then pⱼ = exp(sⱼ − m) per lane, then wave-sum.
Two reductions per 64 keys per head. Write p to LDS (256 B per head).

**Phase C — output, lane ↔ dim.** Now lane *d* owns output dims 2d, 2d+1. It loops over the 64 keys:
*o[d] += pⱼ · V[j][d]*, with pⱼ read from LDS as a broadcast and V[j] read across lanes — 64 lanes ×
4 B = one coalesced 256 B row per key. **128 FMAs per lane per head, zero cross-lane operations.**

Per lane, live state: 2 score accumulators, 2 output accumulators (2 dims × 2 heads = 4 floats),
running m and l per head (4), loop temporaries. **~16 VGPRs.** That is what buys the occupancy in R2.

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

- **Phase A's stride.** Lane↔key means 64 lanes reading rows 256 B apart. It relies on L1 holding
  64 lanes × 64 B = 4 KB of segments across the inner loop ⟨verify L1 is ≥16 KB and the access
  pattern does not thrash it⟩. If it does thrash, the fallback is staging K through LDS with a
  coalesced lane↔dim load and a transpose — 16 KB per block, well within 64 KB — at the cost of a
  barrier. That is the one place the design has a plan B.
- **The reduction count is 6 steps not 5** per wave-max / wave-sum. Two per block per head. Not a
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
