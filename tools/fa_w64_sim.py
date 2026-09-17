#!/usr/bin/env python3
"""
Faithful CPU simulation of kernel_flash_attn_ext_vec_w64 + kernel_flash_attn_ext_vec_reduce_w64.

Models the actual lane -> (key, dim) mapping, the stride arithmetic on a flat buffer laid out the
way the decode path presents it, both shuffle ladders with real simd_shuffle_down semantics, the
threadgroup accumulator ownership, the cross-simdgroup reduce, the interleaved partial layout and
the reduce kernel. Compared against a float64 reference attention.

This cannot prove the kernel runs; it proves the indexing and the reductions are right.
"""
import numpy as np

NW = 64
NE = 2
NL = NW // NE            # 32
C = NW                   # 64 - the binding invariant
FLT_MAX = np.float32(3.4028234663852886e38)
NEG = float(-FLT_MAX / 2)
MAXHALF = 65504.0


def shuffle_down(vals, delta, garbage):
    """simd_shuffle_down over a 64-lane vector. Lanes whose source is out of range get a value
    the kernel must not depend on - pass `garbage` to prove it does not."""
    out = np.empty_like(vals)
    for i in range(NW):
        j = i + delta
        out[i] = vals[j] if j < NW else garbage
    return out


def shuffle(vals, idx):
    return np.array([vals[idx[i]] for i in range(NW)], dtype=vals.dtype)


def run_kernel(Q, Kbuf, Vbuf, maskrow, ne11, ne02, ne12, DK, DV, scale, nsg, nwg,
               garbage, kvpad_pad=None):
    """Q: [ne02, DK] float32 (one query row, batch 1).
       Kbuf/Vbuf: flat float32, element index = t*ns10 + h*DK + d   (ns10 = ne12*DK)
       maskrow: [ne11] float32 additive mask (0 or -inf)
       Returns dst [ne02, DV]."""
    DK4, DV4 = DK // 4, DV // 4
    PV4 = max(DV, 128) // 4
    ns10 = ne12 * DK
    ns20 = ne12 * DV
    has_kvpad = (ne11 % C) != 0

    nrows = ne02                      # ne1*ne2*ne3 with ne01 = ne03 = 1
    # temp buffer: O partials interleaved by workgroup, then (S, M) pairs
    htmp = np.zeros(nrows * DV * nwg + 2 * nrows * nwg, dtype=np.float64)

    for iq2 in range(ne02):           # threadgroup grid y
        ikv2 = iq2 // (ne02 // ne12)
        for iwg in range(nwg):        # threadgroup grid z
            # ---- threadgroup memory
            sq4 = Q[iq2].astype(np.float32).astype(np.float64)   # staged as half4 upstream; f32 here
            ss = [np.zeros(C) for _ in range(nsg)]
            sm = [np.zeros(C) for _ in range(nsg)]
            so4 = [np.zeros(PV4 * 4) for _ in range(nsg)]        # flat, PV4 float4 per simdgroup
            Sreg = [0.0] * nsg
            Mreg = [NEG] * nsg

            for sgitg in range(nsg):
                S, M = 0.0, NEG
                k_base = ikv2 * DK
                v_base = ikv2 * DV
                Ksrc, Vsrc = Kbuf, Vbuf
                pm = maskrow

                ic0 = iwg * nsg + sgitg
                while True:
                    ic = ic0 * C
                    if ic >= ne11:
                        break

                    if has_kvpad and ic + C > ne11:
                        Ksrc, Vsrc, pm = kvpad_pad
                        k_base = ikv2 * DK
                        v_base = ikv2 * DV
                        # pad buffer is [C, ne12, D] per K and per V
                        ic = 0

                    smv = np.array([pm[ic + t] for t in range(C)])
                    sm[sgitg] = smv

                    if smv.max() <= -MAXHALF:
                        ic0 += nwg * nsg
                        continue

                    # ---------------- Q*K^T
                    mqk = np.zeros((C // NE, NW))
                    for lane in range(NW):
                        tx, ty = lane % NL, lane // NL
                        for cc in range(C // NE):
                            acc = 0.0
                            for ii in range(DK4 // NL):
                                off = (ty + cc * NE) * ns10 + 4 * (ii * NL + tx)
                                k4 = Ksrc[k_base + ic * ns10 + off: k_base + ic * ns10 + off + 4]
                                q4 = sq4[4 * (ii * NL + tx): 4 * (ii * NL + tx) + 4]
                                acc += float(np.dot(k4.astype(np.float64), q4))
                            mqk[cc, lane] = acc

                    for cc in range(C // NE):
                        v = mqk[cc].copy()
                        for d in (32, 16, 8, 4, 2, 1):
                            if NL > d:
                                v = v + shuffle_down(v, d, garbage)
                        v = shuffle(v, [NL * (l // NL) for l in range(NW)])
                        mqk[cc] = v

                    for lane in range(NW):
                        tx, ty = lane % NL, lane // NL
                        mv = smv[NE * tx + ty]
                        masked = mv <= -MAXHALF
                        sval = mqk[tx, lane] * scale + mv
                        ss[sgitg][NE * tx + ty] = NEG if masked else sval

                    # ---------------- online softmax
                    m = M
                    svals = ss[sgitg].copy()
                    M = max(M, svals.max())
                    ms = np.exp(m - M)
                    vs = np.exp(svals - M)
                    S = S * ms + vs.sum()
                    ss[sgitg] = vs
                    so4[sgitg] *= ms

                    # ---------------- O += P*V
                    lo = np.zeros((NW, 4))
                    for lane in range(NW):
                        tx, ty = lane % NL, lane // NL
                        for cc in range(C // NE):
                            ps = ss[sgitg][NE * cc + ty]
                            for ii in range(DV4 // NL):
                                off = (ty + cc * NE) * ns20 + 4 * (ii * NL + tx)
                                vv = Vsrc[v_base + ic * ns20 + off: v_base + ic * ns20 + off + 4]
                                lo[lane] += 0.0 if ps == 0.0 else vv.astype(np.float64) * ps

                    for comp in range(4):
                        v = lo[:, comp].copy()
                        for d in (32, 16, 8, 4, 2, 1):
                            if NL <= d < NW:
                                v = v + shuffle_down(v, d, garbage)
                        lo[:, comp] = v

                    for lane in range(NL):           # ty == 0 owns the accumulator
                        so4[sgitg][4 * lane: 4 * lane + 4] += lo[lane]

                    ic0 += nwg * nsg

                Sreg[sgitg], Mreg[sgitg] = S, M
                ss[sgitg] = ss[sgitg].copy()
                ss[sgitg][0], ss[sgitg][1] = S, M

            # ---------------- cross-simdgroup reduce
            r = nsg // 2
            while r > 0:
                for sgitg in range(r):
                    S0, M0 = ss[sgitg][0], ss[sgitg][1]
                    S1, M1 = ss[sgitg + r][0], ss[sgitg + r][1]
                    Mx = max(M0, M1)
                    ms0, ms1 = np.exp(M0 - Mx), np.exp(M1 - Mx)
                    ss[sgitg][0] = S0 * ms0 + S1 * ms1
                    ss[sgitg][1] = Mx
                    so4[sgitg] = so4[sgitg] * ms0 + so4[sgitg + r] * ms1
                r >>= 1

            # ---------------- store interleaved partials (nwg > 1 -> no normalise here)
            rid = iq2
            for i in range(DV4):
                htmp[(rid * DV4 * nwg + nwg * i + iwg) * 4:
                     (rid * DV4 * nwg + nwg * i + iwg) * 4 + 4] = so4[0][4 * i: 4 * i + 4]
            base1 = nrows * DV * nwg
            htmp[base1 + rid * 2 * nwg + 2 * iwg + 0] = ss[0][0]
            htmp[base1 + rid * 2 * nwg + 2 * iwg + 1] = ss[0][1]

    # ---------------- reduce kernel
    DV4 = DV // 4
    out = np.zeros((nrows, DV))
    base1 = nrows * DV * nwg
    for rid in range(nrows):
        Slanes = np.zeros(NW)
        Mlanes = np.full(NW, NEG)
        for lane in range(NW):
            live = lane < nwg
            iwgr = lane if live else 0
            if live:
                Slanes[lane] = htmp[base1 + rid * 2 * nwg + 2 * iwgr + 0]
                Mlanes[lane] = htmp[base1 + rid * 2 * nwg + 2 * iwgr + 1]
        m = Mlanes.max()
        ms = np.array([np.exp(Mlanes[l] - m) if l < nwg else 0.0 for l in range(NW)])
        Ssum = float((Slanes * ms).sum())
        Sinv = 0.0 if Ssum == 0.0 else 1.0 / Ssum
        for i in range(DV4):
            acc = np.zeros(4)
            for lane in range(NW):
                iwgr = lane if lane < nwg else 0
                acc += htmp[(rid * DV4 * nwg + i * nwg + iwgr) * 4:
                            (rid * DV4 * nwg + i * nwg + iwgr) * 4 + 4] * ms[lane]
            out[rid, 4 * i: 4 * i + 4] = acc * Sinv
    return out


def reference(Q, Kbuf, Vbuf, maskrow, ne11, ne02, ne12, DK, DV, scale):
    ns10, ns20 = ne12 * DK, ne12 * DV
    out = np.zeros((ne02, DV))
    for h in range(ne02):
        hkv = h // (ne02 // ne12)
        sc = np.empty(ne11)
        for t in range(ne11):
            k = Kbuf[t * ns10 + hkv * DK: t * ns10 + hkv * DK + DK].astype(np.float64)
            sc[t] = np.dot(k, Q[h].astype(np.float64)) * scale + maskrow[t]
        keep = maskrow > -MAXHALF
        if not keep.any():
            continue
        m = sc[keep].max()
        p = np.where(keep, np.exp(sc - m), 0.0)
        acc = np.zeros(DV)
        for t in range(ne11):
            if p[t] == 0.0:
                continue
            acc += Vbuf[t * ns20 + hkv * DV: t * ns20 + hkv * DV + DV].astype(np.float64) * p[t]
        out[h] = acc / p.sum()
    return out


def make_pad(Kbuf, Vbuf, maskrow, ne11, ne12, DK, DV):
    """What kernel_flash_attn_ext_pad writes: the last ne11 % C rows copied, the rest zeroed,
    mask tail set to -MAXHALF. Laid out so the kernel's ic = 0 addressing hits it."""
    icp = ne11 % C
    ic0 = ne11 - icp
    ns10, ns20 = ne12 * DK, ne12 * DV
    kp = np.zeros(C * ns10, dtype=np.float32)
    vp = np.zeros(C * ns20, dtype=np.float32)
    mp = np.full(C, -MAXHALF)
    for i in range(icp):
        kp[i * ns10:(i + 1) * ns10] = Kbuf[(ic0 + i) * ns10:(ic0 + i + 1) * ns10]
        vp[i * ns20:(i + 1) * ns20] = Vbuf[(ic0 + i) * ns20:(ic0 + i + 1) * ns20]
        mp[i] = maskrow[ic0 + i]
    return kp, vp, mp


def case(ne11, valid, ne02, ne12, nsg, nwg, poison, garbage, seed):
    DK = DV = 128
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(DK)
    Q = rng.standard_normal((ne02, DK)).astype(np.float32)
    ns10 = ne12 * DK
    Kbuf = rng.standard_normal(ne11 * ns10).astype(np.float32)
    Vbuf = rng.standard_normal(ne11 * ns10).astype(np.float32)
    maskrow = np.where(np.arange(ne11) < valid, 0.0, -np.inf)
    if poison:
        # what the over-allocated tail of the decode cache actually looks like
        Kbuf[valid * ns10:] = np.nan
        Vbuf[valid * ns10:] = np.nan

    pad = None
    if ne11 % C:
        pad = make_pad(Kbuf, Vbuf, maskrow, ne11, ne12, DK, DV)

    got = run_kernel(Q, Kbuf, Vbuf, maskrow, ne11, ne02, ne12, DK, DV, scale, nsg, nwg,
                     garbage, pad)
    want = reference(Q, Kbuf, Vbuf, maskrow, ne11, ne02, ne12, DK, DV, scale)
    err = np.abs(got - want).max()
    nmse = ((got - want) ** 2).sum() / max((want ** 2).sum(), 1e-30)
    nan = int(np.isnan(got).sum())
    ok = (not nan) and nmse <= 5e-4 and err < 1e-9
    print("  ne11=%-6d valid=%-6d hq=%-3d hkv=%-3d nsg=%d nwg=%-3d poison=%-5s garbage=%-8s "
          "maxerr=%.3e nmse=%.3e nan=%d %s"
          % (ne11, valid, ne02, ne12, nsg, nwg, poison, garbage, err, nmse, nan,
             "OK" if ok else "*** FAIL ***"))
    return ok


if __name__ == "__main__":
    allok = True
    print("wave64 vec kernel simulation (NW=%d NE=%d NL=%d C=%d)" % (NW, NE, NL, C))
    for garbage in (0.0, 1e9, float("nan")):
        print(" out-of-range shuffle_down source = %s" % garbage)
        for (ne11, valid, hq, hkv, nsg, nwg) in [
            (64,   64,   16, 8, 1, 2),
            (128,  128,  16, 8, 2, 2),
            (128,  70,   16, 8, 1, 4),
            (192,  113,  16, 8, 2, 2),
            (200,  200,  16, 8, 1, 3),
            (200,  137,  16, 8, 2, 3),
            (256,  1,    16, 8, 2, 2),
            (256,  256,  16, 16, 1, 2),   # no GQA
            (256,  200,  4,  1, 2, 2),    # 4:1 GQA
            (320,  289,  16, 8, 4, 2),
            (384,  384,  16, 8, 2, 3),
        ]:
            for poison in (False, True):
                allok &= case(ne11, valid, hq, hkv, nsg, nwg, poison, garbage, seed=ne11 + valid)
    print("ALL OK" if allok else "FAILURES PRESENT")
    raise SystemExit(0 if allok else 1)
