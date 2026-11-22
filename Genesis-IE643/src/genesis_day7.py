"""genesis_day7.py — Genesis (IE 643), novelty assessment: what breaks at the joins, and a targeted fix.

For a chained video (its manifest gives the join positions) we measure, AT EVERY JOIN j:
  appearance jump  = (1 - DINO cos(last frame before, first frame after)) / median of the same quantity between
                     neighbouring frames inside clips.  1 = the join looks like an ordinary frame step.
  identity to start= DINO cos(mean of clip j+1, mean of clip 0): how far the subject has drifted from the start.
  speed ratio      = optical-flow speed in the 6 frame steps after the join / the 6 before it (1 = no hitch).
  colour jump      = LAB colour change across the join / median colour change between neighbouring frames.
A join is flagged when
  appearance break : appearance jump > 3            (subject/scene changes abruptly at the join)
  identity drift   : identity to start < 0.85       (subject no longer looks like the opening subject)
  motion break     : speed ratio < 0.67 or > 1.5    (speed hitch: motion slows/speeds up abruptly at the join)
  colour jump      : colour jump > 3
The FIRST RECURRING FAILURE is the failure type that occurs at >= 2 joins and whose first occurrence comes earliest
(ties -> the one occurring at more joins). The thresholds were fixed before the runs.

Targeted fixes (chosen by a rule written before the runs):
  motion break     -> m = 3 context latents (9 frames) + LoRA: more context frames carry the velocity across the join
  appearance break / identity drift -> generate-and-select: 2 candidates per clip, keep the best by DINO identity
  colour jump      -> (Method C already anchors colour; reported, no extra run)
"""
import json, os

import numpy as np

THRESH = {"appearance": 3.0, "identity": 0.85, "speed_lo": 0.67, "speed_hi": 1.5, "colour": 3.0}
FIX_FOR = {"motion break": "m3", "appearance break": "select", "identity drift": "select", "colour jump": None}


def join_diagnosis(frames_u8, manifest, models, w=6):
    """One row per join of a chained video. models: genesis_metrics.Models (DINO is used)."""
    import genesis_metrics as gm
    import genesis_day6 as g6
    b = [int(x) for x in manifest["boundaries"]]
    N = len(frames_u8)
    dino = models.image_dino(frames_u8).numpy()                      # normalised, N x D
    adj = 1 - (dino[1:] * dino[:-1]).sum(1)                         # adj[t-1] = change from frame t-1 to t
    inside = np.ones(N - 1, bool)
    for s in b:
        inside[s - 1] = False
    base_app = np.median(adj[inside]) + 1e-6
    lab = gm._lab(gm._small(frames_u8))
    mcol = lab.reshape(N, -1, 3).mean(1)
    dcol = np.linalg.norm(mcol[1:] - mcol[:-1], axis=1)
    base_col = np.median(dcol[inside]) + 1e-6
    speed = g6.flow_speeds(frames_u8)
    segs = list(zip([0] + b, b + [N]))
    clip_mean = [dino[a:e].mean(0) / np.linalg.norm(dino[a:e].mean(0)) for a, e in segs]
    rows = []
    for j, s in enumerate(b):
        before = np.nanmean(speed[max(1, s - w):s])
        after = np.nanmean(speed[s + 1:min(N, s + 1 + w)])
        r = {"join": j + 1, "frame": s, "time_s": round(s / manifest.get("fps", 24), 2),
             "appearance_jump": float(adj[s - 1] / base_app),
             "identity_to_start": float(clip_mean[j + 1] @ clip_mean[0]),
             "speed_ratio": float(after / before) if before > 1e-3 else float("nan"),
             "colour_jump": float(dcol[s - 1] / base_col)}
        r["appearance break"] = r["appearance_jump"] > THRESH["appearance"]
        r["identity drift"] = r["identity_to_start"] < THRESH["identity"]
        r["motion break"] = (r["speed_ratio"] < THRESH["speed_lo"]) or (r["speed_ratio"] > THRESH["speed_hi"])
        r["colour jump"] = r["colour_jump"] > THRESH["colour"]
        rows.append(r)
    return rows, speed


def first_recurring_failure(rows):
    """(failure type, first join, count) of the earliest failure type that occurs at >= 2 joins, else (None, ...)."""
    best = None
    for f in FIX_FOR:
        hits = [r["join"] for r in rows if r[f]]
        if len(hits) >= 2:
            key = (hits[0], -len(hits))
            if best is None or key < best[0]:
                best = (key, f, hits[0], len(hits))
    if best:
        return best[1], best[2], best[3]
    once = [(f, [r["join"] for r in rows if r[f]]) for f in FIX_FOR]     # nothing recurs: the earliest single one
    once = sorted([(h[0], f, len(h)) for f, h in once if h])
    return (once[0][1], once[0][0], once[0][2]) if once else (None, None, 0)


def summary(rows):
    a = lambda k: np.array([r[k] for r in rows], float)
    sr = a("speed_ratio")
    return {"joins": len(rows),
            "appearance_jump_median": float(np.nanmedian(a("appearance_jump"))),
            "identity_to_start_last": float(rows[-1]["identity_to_start"]) if rows else float("nan"),
            "speed_abslog2_mean": float(np.nanmean(np.abs(np.log2(np.clip(sr, 1e-3, None))))),
            "colour_jump_median": float(np.nanmedian(a("colour_jump"))),
            **{f"{f} joins": int(sum(r[f] for r in rows)) for f in FIX_FOR}}


def mark_joins(frames_u8, boundaries, label="", hold=6, fps=24):
    """Copy of the video with the clip number on every frame and a red border + 'JOIN j' on the first `hold`
    frames after every join, so the joins are visible when the video plays."""
    from PIL import Image, ImageDraw
    b = [int(x) for x in boundaries]
    out = []
    for t, f in enumerate(frames_u8):
        k = sum(t >= s for s in b)                                    # clip index
        im = Image.fromarray(f.copy()); d = ImageDraw.Draw(im)
        txt = f"{label}  clip {k}  {t / fps:4.1f}s"
        d.rectangle([0, 0, 8 + 7 * len(txt), 18], fill=(0, 0, 0)); d.text((4, 3), txt, fill=(255, 255, 255))
        j = [i for i, s in enumerate(b) if 0 <= t - s < hold]
        if j:
            W, H = im.size
            for q in range(5):
                d.rectangle([q, q, W - 1 - q, H - 1 - q], outline=(230, 30, 30))
            d.rectangle([W - 90, H - 26, W - 6, H - 6], fill=(230, 30, 30)); d.text((W - 84, H - 21), f"JOIN {j[0] + 1}", fill=(255, 255, 255))
        out.append(np.asarray(im))
    return np.stack(out)
