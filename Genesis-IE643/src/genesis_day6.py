"""genesis_day6.py — Genesis (IE 643), novelty stage: tools for 30-60 s videos.

1. Prompt planner (fixes prompt starvation / stalling)
   A story is given as a subject, a style and a list of short BEATS (one visible change each). The planner works out
   how many clips the duration needs and spreads the beats over them in order, so each beat lasts ~2 clips
   (~3.7 s at m = 2), i.e. about one new prompt every 4 s, and warns when the story has too few beats.
   Every prompt repeats the same subject description and style ("the same ...") so identity is re-stated each time.
   Optional beat boost: the first clip of each beat uses a higher CFG scale, so a new prompt is pushed harder
   exactly where it starts (targets the "chained clips stop following new prompts" result).

2. Generate-and-select (fixes bad clips / identity flips)
   Each continuation clip is generated twice (different seeds); CandidateScorer keeps the better one by
       total = DINO identity to the shot's first clip + 2 * CLIP-T(prompt) - 0.05 * min(seam jump, 5)
   (weights fixed before any results: identity dominates, CLIP-T breaks ties, the seam term rejects jumps).
   The scorer runs on the CPU, so the GPU only holds the video model.

3. Seam speed (measures the speed hitch)
   For every seam: mean optical-flow speed in the 6 frame steps after it / the 6 before it (1.0 = no hitch).
   Also the end-of-clip speed-up: speed in the last quarter of each clip / the rest of that clip.

4. Held-out continuation benchmark (direct test of the LoRA)
   Real OpenVid clips that were NOT among the 300 LoRA training clips: the model gets the first K real frames and the
   clip's caption, generates the rest, and we compare with what really happened (LPIPS, DINO, colour, motion ratio).
"""
import json, math, os

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ 1. planner
def plan_script(subject, style, beats, seconds, fps, num_frames, m, opening=None, beat_cfg=None, key_prefix="s",
                template="The same {subject} {beat}, {style}"):
    """Returns (items, info). items are run_script items: prompt, key (one embedding per beat), cut=False,
    beat index and, with beat_cfg, a per-clip 'cfg' for the first clip of every beat after the first.
    opening: full first prompt (default: 'A {subject} {beat0}, {style}'); template: every later prompt."""
    import genesis_lib as gl
    n = gl.clips_for_duration(seconds, fps, num_frames, m)
    prompts = []
    for j, b in enumerate(beats):
        if j == 0:
            prompts.append(opening or f"A {subject} {b}, {style}")
        else:
            prompts.append(template.format(subject=subject, beat=b, style=style))
    base = [{"prompt": p, "cut": False, "beat": j, "key": f"{key_prefix}_b{j}"} for j, p in enumerate(prompts)]
    if n < len(base):
        base = base[:n]
    reps = [n // len(base) + (1 if i < n % len(base) else 0) for i in range(len(base))]
    items = []
    for it, r in zip(base, reps):
        for k in range(r):
            x = dict(it)
            if beat_cfg is not None and k == 0 and it["beat"] > 0:
                x["cfg"] = float(beat_cfg)
            items.append(x)
    clip_sec = gl.new_frames_per_clip(num_frames, m) / fps
    sec_per_beat = n * clip_sec / len(base)
    info = {"clips": n, "beats": len(base), "sec_per_beat": round(sec_per_beat, 2),
            "seconds": round((num_frames + (n - 1) * gl.new_frames_per_clip(num_frames, m)) / fps, 2),
            "warning": (f"only {len(base)} beats for {seconds} s ({sec_per_beat:.1f} s each): the video will stall; "
                        f"add beats (about one per 4 s)") if sec_per_beat > 4.5 else None}
    return items, info


def prompts_needed(items):
    """{embedding key: prompt} for the text encoder."""
    return {it["key"]: it["prompt"] for it in items}


# ------------------------------------------------------------------------------------------------ 2. scorer
class CandidateScorer:
    """Scores a candidate continuation clip (CPU, fp32). frames: float [0,1] F x H x W x 3 incl. the K context
    frames at the start; ctx: uint8 context frames (the previous clip's tail); first: float frames of the shot's
    first clip."""

    W_CLIP, W_SEAM, SEAM_CAP = 2.0, 0.05, 5.0

    def __init__(self, clip_id="openai/clip-vit-large-patch14", dino_id="facebook/dino-vitb16", device="cpu",
                 n_frames=6):
        import genesis_metrics as gm
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.M = gm.Models(device=device, clip_id=clip_id, dino_id=dino_id, dtype=torch.float32)
        self.n = n_frames
        self._shot = {}

    def _pick(self, frames):
        idx = np.unique(np.linspace(0, len(frames) - 1, self.n).round().astype(int))
        return (np.clip(frames[idx], 0, 1) * 255).round().astype(np.uint8)

    def score(self, frames, K, ctx, first, prompt, shot_key=None):
        import genesis_metrics as gm
        new = frames[K:]
        if shot_key not in self._shot:
            d = self.M.image_dino(self._pick(first))
            self._shot = {shot_key: F.normalize(d.mean(0), dim=0)}
        ref = self._shot[shot_key]
        u8 = self._pick(new)
        d = self.M.image_dino(u8)
        ident = float(F.normalize(d.mean(0), dim=0) @ ref)
        clipt = float((self.M.image_clip(u8) @ self.M.text_clip([prompt])[0]).mean()) if prompt else 0.0
        # seam: jump from the last context frame to the first new frame, relative to the clip's own frame steps
        sm = gm._small((np.clip(np.concatenate([ctx[-1:].astype(np.float32) / 255.0, new]), 0, 1) * 255)
                       .round().astype(np.uint8)).astype(np.float32) / 255.0
        steps = np.abs(sm[1:] - sm[:-1]).mean(axis=(1, 2, 3))
        seam = float(steps[0] / (np.median(steps[1:]) + 1e-6)) if len(steps) > 1 else 1.0
        total = ident + self.W_CLIP * clipt - self.W_SEAM * min(seam, self.SEAM_CAP)
        return {"identity": round(ident, 4), "clip_t": round(clipt, 4), "seam": round(seam, 3), "total": round(total, 4)}


# ------------------------------------------------------------------------------------------------ 3. seam speed
def flow_speeds(frames_u8, width=208):
    """speed[t] = mean Farneback flow magnitude between frame t-1 and t (pixels/frame at full resolution)."""
    import cv2
    import genesis_metrics as gm
    small = gm._small(frames_u8, width)
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in small]
    scale = frames_u8.shape[2] / small.shape[2]
    sp = np.full(len(grey), np.nan)
    for t in range(1, len(grey)):
        fl = cv2.calcOpticalFlowFarneback(grey[t - 1], grey[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        sp[t] = float(np.linalg.norm(fl, axis=-1).mean() * scale)
    return sp


def seam_speed(speeds, boundaries, cuts=(), w=6):
    """Per-seam speed ratio after/before (excluding the seam step itself and intentional cuts) and the
    end-of-clip speed-up (last quarter of a clip / its first three quarters)."""
    N = len(speeds)
    b = [int(x) for x in boundaries if 0 < int(x) < N and int(x) not in set(int(c) for c in cuts)]
    ratios = []
    for s in b:
        before = np.nanmean(speeds[max(1, s - w):s])
        after = np.nanmean(speeds[s + 1:min(N, s + 1 + w)])
        if before > 1e-3 and np.isfinite(after):
            ratios.append(float(after / before))
    segs = list(zip([0] + [int(x) for x in boundaries], [int(x) for x in boundaries] + [N]))
    ends = []
    for a, e in segs:
        L = e - a
        if L >= 12:
            q = a + int(0.75 * L)
            body, tail = np.nanmean(speeds[a + 1:q]), np.nanmean(speeds[q:e])
            if body > 1e-3:
                ends.append(float(tail / body))
    lr = [abs(math.log2(r)) for r in ratios if r > 0]
    return {"seam_speed_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "seam_speed_abslog2": float(np.mean(lr)) if lr else float("nan"),
            "seam_speed_drops": f"{sum(r < 0.9 for r in ratios)}/{len(ratios)}",
            "seam_speed_jumps": f"{sum(r > 1.1 for r in ratios)}/{len(ratios)}",
            "end_speedup": float(np.mean(ends)) if ends else float("nan"),
            "ratios": [round(r, 3) for r in ratios]}


# ------------------------------------------------------------------------------------------------ 4. held-out
def load_clip_from_file(path, n, H, W, target_fps=24):
    """Same frame sampling as the LoRA training data (genesis_train_lora.load_clip): ~24 fps, cover-resize,
    centre crop, first n frames. Returns uint8 n x H x W x 3 or None."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(fps / target_fps)))
    frames, idx = [], 0
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = fr.shape[:2]
            s = max(H / h, W / w)
            fr = cv2.resize(fr, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA)
            y, x = (fr.shape[0] - H) // 2, (fr.shape[1] - W) // 2
            frames.append(cv2.cvtColor(fr[y:y + H, x:x + W], cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return np.stack(frames) if len(frames) == n else None


def run_heldout(G, E, out, clips, conds, lora_dir, log_path, num_frames=33, cfg=3.5):
    """clips: [{'id', 'key', 'seed'}]; real frames in {out}/heldout/real/{id}.npy. conds: [{'name', 'm', 'lora'}].
    Saves each generated clip as {out}/heldout/gen/{cond}__{id}.mp4 (all frames, incl. the K context frames)."""
    import time
    import genesis_lib as gl
    from diffusers.utils import export_to_video
    os.makedirs(f"{out}/heldout/gen", exist_ok=True)
    nf0 = G.nf
    G.nf = num_frames
    try:
        for cond in conds:
            G.set_lora(lora_dir if cond["lora"] else None)
            K = gl.context_frames(cond["m"])
            for c in clips:
                p = f"{out}/heldout/gen/{cond['name']}__{c['id']}.mp4"
                if os.path.exists(p):
                    continue
                real = np.load(f"{out}/heldout/real/{c['id']}.npy")
                t0 = time.time()
                frames, _ = G.context_clip(real[:K], E, c["key"], c["seed"], guidance=cfg)
                export_to_video(list(frames), p, fps=G.fps)
                gl.log_jsonl(log_path, {"heldout": cond["name"], "clip": c["id"], "sec": round(time.time() - t0, 1),
                                        "frames_ok": gl.frames_ok(frames)})
    finally:
        G.nf = nf0
        G.set_lora(None)


def eval_heldout(out, conds, device="cuda", models=None):
    """Compares every generated continuation with the real one (frames K..end only).
    lpips (AlexNet, lower = closer), dino (cos per frame, higher = closer), dE (LAB mean-colour distance),
    motion_ratio (generated speed / real speed; 1 = same speed)."""
    import pandas as pd
    import genesis_lib as gl
    import genesis_metrics as gm
    import lpips
    loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    if models is None:
        models = gm.Models(device=device)
    rows = []
    for cond in conds:
        K = gl.context_frames(cond["m"])
        for p in sorted(os.listdir(f"{out}/heldout/gen")):
            if not p.startswith(cond["name"] + "__"):
                continue
            cid = p[len(cond["name"]) + 2:-4]
            real = np.load(f"{out}/heldout/real/{cid}.npy")
            gen = gm.read_u8(f"{out}/heldout/gen/{p}")
            n = min(len(real), len(gen))
            r, g = real[K:n], gen[K:n]
            def t(x):
                return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
            with torch.no_grad():
                lp = float(torch.cat([loss(t(g[i:i + 8]), t(r[i:i + 8])).flatten() for i in range(0, len(g), 8)]).mean())
            dino = float((models.image_dino(g) * models.image_dino(r)).sum(-1).mean())
            lab_r, lab_g = gm._lab(gm._small(r)), gm._lab(gm._small(g))
            dE = float(np.linalg.norm(lab_r.reshape(-1, 3).mean(0) - lab_g.reshape(-1, 3).mean(0)))
            sp_r = np.nanmean(flow_speeds(np.concatenate([real[K - 1:K], r])))
            sp_g = np.nanmean(flow_speeds(np.concatenate([gen[K - 1:K], g])))
            rows.append({"cond": cond["name"], "m": cond["m"], "lora": cond["lora"], "clip": cid, "lpips": lp,
                         "dino": dino, "dE": dE, "motion_real": sp_r, "motion_gen": sp_g,
                         "motion_ratio": sp_g / max(sp_r, 1e-3)})
    return pd.DataFrame(rows)


def paired_summary(df):
    """Mean of each metric per condition, plus per-clip win rate of LoRA vs no LoRA at the same m."""
    import pandas as pd
    df = df.copy()
    df["motion_err"] = (np.log2(df["motion_ratio"].clip(lower=1e-3))).abs()
    agg = df.groupby("cond")[["lpips", "dino", "dE", "motion_ratio", "motion_err"]].agg(["mean", "std"]).round(4)
    wins = []
    for m in sorted(df["m"].unique()):
        a = df[(df["m"] == m) & (~df["lora"])].set_index("clip")
        b = df[(df["m"] == m) & (df["lora"])].set_index("clip")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        wins.append({"m": int(m), "clips": len(common),
                     "LoRA better LPIPS": f"{int((b.loc[common, 'lpips'] < a.loc[common, 'lpips']).sum())}/{len(common)}",
                     "LoRA better DINO": f"{int((b.loc[common, 'dino'] > a.loc[common, 'dino']).sum())}/{len(common)}",
                     "LoRA closer speed": f"{int((b.loc[common, 'motion_err'] < a.loc[common, 'motion_err']).sum())}/{len(common)}"})
    return agg, pd.DataFrame(wins)
