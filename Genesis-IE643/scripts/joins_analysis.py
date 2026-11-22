"""Novelty assessment: measure every join of the 20 s fox video (Method C vs C + LoRA), find the first recurring
failure, apply the colour hand-off fix, and write tables, figures and a joins-marked video.
Usage: python scripts/joins_analysis.py --day5 path/to/genesis5 --out results/novelty"""
import json, os, sys, numpy as np, pandas as pd, cv2, imageio.v3 as iio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import genesis_day6 as g6, genesis_day7 as g7, genesis_metrics as gm
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import argparse, os
ap = argparse.ArgumentParser(description="Per-join diagnosis of the Day 5 20 s fox videos (C vs C + LoRA) and the colour hand-off fix.")
ap.add_argument("--day5", required=True, help="folder genesis5/ from the Day 5 output (videos/ and manifests/)")
ap.add_argument("--out", default="results/novelty")
a = ap.parse_args(); B = a.day5; FPS = 24
os.makedirs(a.out, exist_ok=True); os.chdir(a.out)
LAB = {"C_anchor": "C", "C_anchor_lora": "C + LoRA"}; COL = {"C_anchor": "#8a8985", "C_anchor_lora": "#2a6fdb"}
def handoff(v, bounds, T=12):
    """Colour hand-off at joins: the first frame after a join is matched (LAB mean and spread) to the last frame before it,
    and the correction fades out linearly over T frames (0.5 s), so the clip's own (anchored) colour takes over smoothly."""
    out = v.copy()
    tolab = lambda f: cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    for s in bounds:
        p, q = tolab(out[s - 1]).reshape(-1, 3), tolab(out[s]).reshape(-1, 3)
        mp, sp_, mq, sq = p.mean(0), p.std(0) + 1e-6, q.mean(0), q.std(0) + 1e-6
        for t in range(T):
            if s + t >= len(out): break
            w = 1 - t / T
            f = tolab(out[s + t])
            f = (f - mq) * (1 + w * (sp_ / sq - 1)) + mq + w * (mp - mq)
            out[s + t] = (np.clip(cv2.cvtColor(f, cv2.COLOR_LAB2RGB), 0, 1) * 255).round().astype(np.uint8)
    return out
LAB["C_anchor_lora_handoff"] = "C + LoRA + colour hand-off"; COL["C_anchor_lora_handoff"] = "#d9572b"
V, M, R, SP = {}, {}, {}, {}
for k in LAB:
    if k == "C_anchor_lora_handoff":
        M[k] = M["C_anchor_lora"]; V[k] = handoff(V["C_anchor_lora"], M[k]["boundaries"])
        iio.imwrite("fox20_C_anchor_lora_handoff_FULL.mp4", V[k], fps=FPS, codec="libx264", quality=8)
    else:
        M[k] = json.load(open(f"{B}/manifests/fox20b_{k}.json")); V[k] = gm.read_u8(f"{B}/videos/fox20b_{k}_FULL.mp4")
    v, b = V[k], M[k]["boundaries"]; N = len(v)
    sm = gm._small(v).astype(np.float32) / 255
    pix = np.abs(sm[1:] - sm[:-1]).mean(axis=(1, 2, 3))
    inside = np.ones(N - 1, bool); inside[[s - 1 for s in b]] = False
    lab = gm._lab(gm._small(v)).reshape(N, -1, 3).mean(1); dcol = np.linalg.norm(lab[1:] - lab[:-1], axis=1)
    sp = g6.flow_speeds(v); SP[k] = sp
    segs = list(zip([0] + b, b + [N]))
    clipspeed = [np.nanmean(sp[a + 1:e]) for a, e in segs]
    rows = []
    for j, s in enumerate(b):
        bef, aft = np.nanmean(sp[s - 6:s]), np.nanmean(sp[s + 1:s + 7])
        a, e = segs[j]; L = e - a; q = a + int(0.75 * L)
        rows.append({"join": j + 1, "time_s": round(s / FPS, 2), "pixel_jump": pix[s - 1] / np.median(pix[inside]),
                     "speed_ratio": aft / bef, "colour_jump": dcol[s - 1] / np.median(dcol[inside]),
                     "end_speedup": np.nanmean(sp[q:e]) / np.nanmean(sp[a + 1:q]),
                     "lab_L_from_start": float(lab[s:segs[j + 1][1]].mean(0)[0] - lab[:49].mean(0)[0])})
    d = pd.DataFrame(rows)
    d["motion_break"] = (d.speed_ratio < 0.67) | (d.speed_ratio > 1.5)
    d["appearance_break"] = d.pixel_jump > 3
    d["colour_jump_flag"] = d.colour_jump > 3
    R[k] = d; d.round(3).to_csv(f"joins_{k}.csv", index=False)
    print(LAB[k]); print(d.round(2).to_string(index=False)); print("clip speeds", np.round(clipspeed, 2))
summ = []
for k, d in R.items():
    sr = d.speed_ratio.values
    first = {f: (int(d[d[f]]["join"].min()) if d[f].any() else None, int(d[f].sum())) for f in ["motion_break", "appearance_break", "colour_jump_flag"]}
    summ.append({"run": LAB[k], "median_pixel_jump": d.pixel_jump.median(), "mean_abslog2_speed": np.mean(np.abs(np.log2(sr))),
                 "speed_drops(<0.9)": int((sr < 0.9).sum()), "speed_rises(>1.1)": int((sr > 1.1).sum()),
                 "min_ratio": sr.min(), "max_ratio": sr.max(), "mean_end_speedup": d.end_speedup.mean(), **{f"{f} (first, n)": v for f, v in first.items()}})
for r, k in zip(summ, R):
    labv = gm._lab(gm._small(V[k])).reshape(len(V[k]), -1, 3).mean(1)
    last = M[k]["boundaries"][-1]
    r["dE_last_clip_vs_first"] = float(np.linalg.norm(labv[last:].mean(0) - labv[:49].mean(0)))
    r["overexp_last_%"] = float((V[k][last:].min(-1) > 245).mean() * 100)
S = pd.DataFrame(summ); print(S.round(3).to_string(index=False)); S.round(3).to_csv("joins_summary.csv", index=False)
# curves
fig, axs = plt.subplots(1, 3, figsize=(15, 3.3))
for a, (key, title, th) in zip(axs, [("pixel_jump", "pixel jump at join (1 = normal frame step)", [3]),
                                    ("speed_ratio", "motion speed after / before join", [0.67, 1.5]),
                                    ("colour_jump", "colour jump at join (1 = normal)", [3])]):
    for k, d in R.items():
        a.plot(d.time_s, d[key], "o-", ms=4, lw=1.6, color=COL[k], label=LAB[k])
    for t in th: a.axhline(t, ls="--", color="#c33", lw=1)
    if key == "speed_ratio": a.axhline(1, color="#999", lw=0.8)
    a.set_title(title, fontsize=10); a.set_xlabel("join time (s)"); a.grid(alpha=0.3)
axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("joins_curves.png", dpi=170); plt.close(fig)
# join frames: C and C+LoRA, before/after for each join
def lab_(img, t):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 8 + 6 * len(t), 14], fill=(0, 0, 0)); dr.text((3, 2), t, fill=(255, 255, 255)); return np.asarray(im)
def sm_(f, s=4): return np.asarray(Image.fromarray(f).resize((f.shape[1] // s, f.shape[0] // s)))
rows = []
for k in LAB:
    for side, off in (("before", -1), ("after", 0)):
        rows.append(np.concatenate([lab_(sm_(V[k][s + off]), f"{LAB[k]} J{j+1} {side}") for j, s in enumerate(M[k]["boundaries"])], 1))
Image.fromarray(np.concatenate(rows, 0)).save("join_frames.jpg", quality=92)
# joins-marked side-by-side video
MK = {k: g7.mark_joins(V[k], M[k]["boundaries"], LAB[k], fps=FPS) for k in LAB}
L = min(len(v) for v in MK.values())
both = np.concatenate([np.stack([sm_(f, 2) for f in MK[k][:L]]) for k in LAB], 2)
iio.imwrite("fox20_ALL_RUNS_JOINS_MARKED.mp4", both, fps=FPS, codec="libx264", quality=8)
# speed timeline
fig, ax = plt.subplots(figsize=(12, 3))
for k in LAB:
    ax.plot(np.arange(len(SP[k])) / FPS, pd.Series(SP[k]).rolling(5, center=True, min_periods=1).mean(), color=COL[k], lw=1.4, label=LAB[k])
for b in M["C_anchor"]["boundaries"]: ax.axvline(b / FPS, color="#bbb", lw=0.7, ls=":")
ax.set_xlabel("time (s) — dotted = joins"); ax.set_ylabel("motion (px/frame)"); ax.legend(); fig.tight_layout(); fig.savefig("speed_timeline.png", dpi=170)
print("done")
