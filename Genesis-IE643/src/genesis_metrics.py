"""genesis_metrics.py — quantitative evaluation for Genesis (IE 643): long videos made by chaining short clips.

Every stitched video comes with a manifest (clip lengths, clip boundaries, intentional [CUT] positions) and the
prompt of every clip. For each video we measure:

  Prompt adherence
    clipsim        CLIP-T: mean cosine(CLIP image feature of each frame, CLIP text feature of that clip's prompt) x 100
    prompt_top1    share of clips whose frames are closest (CLIP) to their OWN prompt among the video's prompts
    prompt_progress  mean over clips i>=1 of cos(clip_i, prompt_i) - cos(clip_i, prompt_0)  (x 100):
                   > 0 means the video moves on to the later prompts instead of staying on the first one
  Temporal consistency (VBench definitions, computed inside each shot)
    subject_cons   mean_t 0.5 [cos(d_t, d_0) + cos(d_t, d_{t-1})], d = DINO ViT-B/16 CLS feature
    background_cons  the same with CLIP image features
    start_end_id   cos(mean DINO feature of the first clip, mean DINO feature of the last clip), per shot
  Transition smoothness
    seam_pix / seam_dino  mean frame-to-frame change AT the clip seams divided by the median change INSIDE clips
                   (pixel L1 / DINO 1-cos). 1.0 = a seam looks like any other frame step; a hard cut is >> 1.
                   Intentional [CUT]s are excluded (reported separately as cut_pix).
    seam_pct       percentile rank of the seam's pixel change among all within-clip frame changes
                   (~50 = an ordinary frame step, 100 = larger than every frame step inside the clips)
  Drift and video quality
    deltaE_final / deltaE_max  CIE-LAB distance between the mean colour of a clip and of the first clip of its shot
    overexp_final  % of pixels with L* > 95 in the last clip (blown-out highlights; Method A's typical failure)
    clipiqa / clipiqa_first / clipiqa_last  CLIP-IQA style zero-shot quality (antonym prompts, 0..1)
    sharpness      variance of the Laplacian (higher = more fine detail)
    motion         mean optical-flow magnitude (Farneback), pixels per frame at full resolution

Note: Method C pulls each frame's LAB mean/std toward the first clip, so deltaE is partly what C optimises.
overexp, clipiqa, start_end_id and the DINO/CLIP consistencies are independent of that correction.

Usage: python genesis_metrics.py --roots DIR [DIR ...] --out OUT_DIR [--device cuda]
Finds every logs/day*_plan.json under the roots; results are cached per video, so re-running only scores new videos.
"""
import argparse, glob, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

METHOD_ORDER = ["baseline", "A_lastframe", "B_m2", "B_m3", "C_anchor", "B_m2_lora", "C_anchor_lora"]
METHOD_COLOR = {"baseline": "#2a78d6", "A_lastframe": "#eb6834", "B_m2": "#1baf7a", "B_m3": "#eda100",
                "C_anchor": "#e87ba4", "B_m2_lora": "#008300", "C_anchor_lora": "#6250d6"}
METHOD_MARK = {"baseline": "o", "A_lastframe": "s", "B_m2": "^", "B_m3": "v", "C_anchor": "D",
               "B_m2_lora": "P", "C_anchor_lora": "X"}
OTHER_COLOR = "#8a8985"
IQA_PAIRS = [("Good photo.", "Bad photo."), ("Sharp photo.", "Blurry photo."),
             ("A photo with natural colors and exposure.", "An overexposed photo with washed-out colors.")]

# column -> (label, better direction) ; direction: +1 higher is better, -1 lower is better, 0 neither
COLUMNS = {
    "clipsim": ("CLIP-T x100", +1), "prompt_top1": ("Prompt top-1", +1), "prompt_progress": ("Prompt progress", +1),
    "subject_cons": ("Subject cons. (DINO)", +1), "background_cons": ("Background cons. (CLIP)", +1),
    "start_end_id": ("Start-end identity", +1), "seam_pix": ("Seam jump (pixel)", -1),
    "seam_dino": ("Seam jump (DINO)", -1), "seam_pct": ("Seam percentile", -1), "deltaE_final": ("Colour drift dE (last clip)", -1),
    "overexp_final": ("Overexposed % (last clip)", -1), "clipiqa": ("CLIP-IQA", +1),
    "clipiqa_drop": ("CLIP-IQA drop first->last", -1), "sharpness": ("Sharpness", +1),
    "motion": ("Motion (px/frame)", 0),
}


# ------------------------------------------------------------------------------------------------ models
class Models:
    """CLIP ViT-L/14 and DINO ViT-B/16 feature extractors (fp16 on GPU)."""

    def __init__(self, device="cuda", clip_id="openai/clip-vit-large-patch14", dino_id="facebook/dino-vitb16",
                 dtype=None):
        from transformers import AutoModel, CLIPModel, CLIPTokenizer
        self.device = device
        self.dtype = dtype or (torch.float16 if str(device).startswith("cuda") else torch.float32)
        def load(cls, mid):
            try:
                return cls.from_pretrained(mid, dtype=self.dtype)            # transformers >= 4.56
            except TypeError:
                return cls.from_pretrained(mid, torch_dtype=self.dtype)
        self.clip = load(CLIPModel, clip_id).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(clip_id)
        self.dino = load(AutoModel, dino_id).to(device).eval()
        self.clip_size = self.clip.config.vision_config.image_size
        self.dino_size = getattr(self.dino.config, "image_size", 224)
        self.clip_mean, self.clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.dino_mean, self.dino_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def _prep(self, frames_u8, size, mean, std):
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        s = size / min(h, w)
        x = F.interpolate(x, size=(max(size, round(h * s)), max(size, round(w * s))), mode="bicubic",
                          align_corners=False, antialias=True)
        h, w = x.shape[-2:]
        y0, x0 = (h - size) // 2, (w - size) // 2
        x = x[:, :, y0:y0 + size, x0:x0 + size]
        m = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        sd = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        return ((x - m) / sd).to(self.dtype)

    @staticmethod
    def _tensor(out, *names):
        if torch.is_tensor(out):
            return out
        for n in names:
            v = getattr(out, n, None)
            if v is not None:
                return v
        raise TypeError(f"unexpected model output {type(out)}")

    @torch.no_grad()
    def image_clip(self, frames_u8, bs=32):
        outs = []
        for i in range(0, len(frames_u8), bs):
            px = self._prep(frames_u8[i:i + bs], self.clip_size, self.clip_mean, self.clip_std)
            f = self._tensor(self.clip.get_image_features(pixel_values=px), "image_embeds", "pooler_output")
            outs.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def image_dino(self, frames_u8, bs=32):
        outs = []
        for i in range(0, len(frames_u8), bs):
            px = self._prep(frames_u8[i:i + bs], self.dino_size, self.dino_mean, self.dino_std)
            f = self.dino(pixel_values=px).last_hidden_state[:, 0]
            outs.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def text_clip(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        f = self._tensor(self.clip.get_text_features(**enc), "text_embeds", "pooler_output")
        return F.normalize(f.float(), dim=-1).cpu()


# ------------------------------------------------------------------------------------------------ helpers
def read_u8(path):
    import imageio.v3 as iio
    return np.stack(list(iio.imiter(path)))


def _small(frames_u8, width=208):
    import cv2
    h, w = frames_u8.shape[1:3]
    size = (width, max(8, int(round(h * width / w))))
    return np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames_u8])


def _lab(frames_small_u8):
    import cv2
    return np.stack([cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB) for f in frames_small_u8])


def _vbench_consistency(feat):
    """VBench subject/background consistency for one shot: mean_t 0.5 [cos(f_t, f_0) + cos(f_t, f_{t-1})]."""
    if feat.shape[0] < 2:
        return float("nan")
    first = (feat[1:] @ feat[0]).clamp(min=0)
    prev = (feat[1:] * feat[:-1]).sum(-1).clamp(min=0)
    return float((0.5 * (first + prev)).mean())


def _segments(n_frames, boundaries):
    b = [x for x in boundaries if 0 < x < n_frames]
    starts = [0] + b
    ends = b + [n_frames]
    return list(zip(starts, ends))


def _nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


# ------------------------------------------------------------------------------------------------ one video
def analyse_video(path, manifest, prompts, models, iqa_text=None):
    """Score one stitched video. Returns (video_record, [clip_records])."""
    import cv2
    t0 = time.time()
    fr = read_u8(path)
    N = fr.shape[0]
    fps = manifest.get("fps", 24)
    bounds = [int(b) for b in manifest.get("boundaries", []) if 0 < int(b) < N]
    cuts = set(int(c) for c in manifest.get("cuts", []))
    segs = _segments(N, bounds)
    n_clips = len(segs)
    prompts = list(prompts)[:n_clips] + [prompts[-1]] * max(0, n_clips - len(prompts)) if prompts else [""] * n_clips

    dino = models.image_dino(fr)
    clipf = models.image_clip(fr)
    small = _small(fr)
    lab = _lab(small)

    # ---- shots (split at intentional cuts) and per-clip shot reference
    shot_of_clip, shot = [], 0
    for i, (a, _) in enumerate(segs):
        if i > 0 and a in cuts:
            shot += 1
        shot_of_clip.append(shot)
    shots = {}
    for i, (a, b) in enumerate(segs):
        s = shot_of_clip[i]
        shots[s] = (shots[s][0], b) if s in shots else (a, b)

    # ---- temporal consistency (inside shots, frame-weighted)
    def per_shot(feat):
        vals, wts = [], []
        for a, b in shots.values():
            v = _vbench_consistency(feat[a:b])
            if not math.isnan(v):
                vals.append(v); wts.append(b - a)
        return float(np.average(vals, weights=wts)) if vals else float("nan")
    subject_cons, background_cons = per_shot(dino), per_shot(clipf)
    # first vs last clip of each shot (comparing across an intentional [CUT] would be meaningless)
    ids = []
    for sh in sorted(set(shot_of_clip)):
        cl = [i for i in range(n_clips) if shot_of_clip[i] == sh]
        if len(cl) >= 2:
            (a0, b0), (aL, bL) = segs[cl[0]], segs[cl[-1]]
            ids.append(float(F.normalize(dino[a0:b0].mean(0), dim=0) @ F.normalize(dino[aL:bL].mean(0), dim=0)))
    start_end_id = float(np.mean(ids)) if ids else float("nan")

    # ---- transition smoothness
    sm = small.astype(np.float32) / 255.0
    d_pix = np.concatenate([[np.nan], np.abs(sm[1:] - sm[:-1]).mean(axis=(1, 2, 3))])
    d_dino = np.concatenate([[np.nan], (1 - (dino[1:] * dino[:-1]).sum(-1)).numpy()])
    inside = np.array([t for t in range(1, N) if t not in set(bounds)], dtype=int)
    trans = [b for b in bounds if b not in cuts]
    med_pix = float(np.median(d_pix[inside])) + 1e-6 if len(inside) else float("nan")
    med_dino = float(np.median(d_dino[inside])) + 1e-6 if len(inside) else float("nan")
    seam_pix = float(np.mean(d_pix[trans]) / med_pix) if trans else float("nan")
    seam_dino = float(np.mean(d_dino[trans]) / med_dino) if trans else float("nan")
    # percentile rank of each seam's pixel change among all within-clip changes (50 = ordinary step, 100 = biggest)
    seam_pct = (float(np.mean([(d_pix[inside] < d_pix[b]).mean() * 100 for b in trans]))
                if trans and len(inside) else float("nan"))
    cut_pix = float(np.mean(d_pix[sorted(cuts & set(bounds))]) / med_pix) if (cuts & set(bounds)) else float("nan")

    # ---- motion (optical flow on the small frames, pairs inside clips, every 2nd frame)
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in small]
    scale = fr.shape[2] / small.shape[2]
    bset = set(bounds)
    flow_mag = {}
    for t in range(1, N, 2):
        if t in bset:
            continue
        flow = cv2.calcOpticalFlowFarneback(grey[t - 1], grey[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mag[t] = float(np.linalg.norm(flow, axis=-1).mean() * scale)
    motion = float(np.mean(list(flow_mag.values()))) if flow_mag else float("nan")

    # ---- prompt adherence
    T = models.text_clip(prompts)                                  # n_clips x D
    uniq = list(dict.fromkeys(prompts))
    TU = models.text_clip(uniq)
    clip_recs, top1, progress = [], [], []
    if iqa_text is None:
        iqa_text = models.text_clip([p for pair in IQA_PAIRS for p in pair])
    logit_scale = 100.0
    for i, (a, b) in enumerate(segs):
        f = clipf[a:b]
        cs = float((f @ T[i]).mean() * 100)
        mf = F.normalize(f.mean(0), dim=0)
        top1.append(float(uniq[int(torch.argmax(TU @ mf))] == prompts[i]))
        if i > 0 and prompts[i] != prompts[0]:
            progress.append(float((f @ T[i]).mean() - (f @ T[0]).mean()) * 100)
        # quality: CLIP-IQA style antonym prompts on every 4th frame
        sims = f[::4] @ iqa_text.T * logit_scale                     # k x (2*pairs)
        sims = sims.view(sims.shape[0], -1, 2)
        iqa = float(torch.softmax(sims, dim=-1)[..., 0].mean())
        # colour drift vs first clip of the same shot
        ref_clip = next(j for j in range(n_clips) if shot_of_clip[j] == shot_of_clip[i])
        ra, rb = segs[ref_clip]
        mu = lab[a:b].reshape(-1, 3).mean(0)
        mu_ref = lab[ra:rb].reshape(-1, 3).mean(0)
        L = lab[a:b, ..., 0]
        chroma = np.sqrt(lab[a:b, ..., 1] ** 2 + lab[a:b, ..., 2] ** 2)
        g_full = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in fr[a:b:8]]
        sharp = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in g_full]))
        mags_i = [v for t, v in flow_mag.items() if a < t < b]
        clip_recs.append({"clip": i, "t_start": round(a / fps, 2), "frames": int(b - a), "shot": shot_of_clip[i],
                          "prompt": prompts[i], "clipsim": cs, "clipiqa": iqa,
                          "deltaE": float(np.linalg.norm(mu - mu_ref)), "L": float(L.mean()),
                          "chroma": float(chroma.mean()), "overexp": float((L > 95).mean() * 100),
                          "underexp": float((L < 5).mean() * 100), "sharpness": sharp,
                          "motion": float(np.mean(mags_i)) if mags_i else float("nan")})

    rec = {"frames": int(N), "seconds": round(N / fps, 2), "n_clips": n_clips, "n_shots": len(shots),
           "clipsim": _nanmean([c["clipsim"] for c in clip_recs]),
           "prompt_top1": _nanmean(top1), "prompt_progress": _nanmean(progress),
           "subject_cons": subject_cons, "background_cons": background_cons, "start_end_id": start_end_id,
           "seam_pix": seam_pix, "seam_dino": seam_dino, "seam_pct": seam_pct, "cut_pix": cut_pix, "motion": motion,
           "deltaE_final": clip_recs[-1]["deltaE"], "deltaE_max": max(c["deltaE"] for c in clip_recs),
           "overexp_final": clip_recs[-1]["overexp"], "overexp_mean": _nanmean([c["overexp"] for c in clip_recs]),
           "clipiqa": _nanmean([c["clipiqa"] for c in clip_recs]), "clipiqa_first": clip_recs[0]["clipiqa"],
           "clipiqa_last": clip_recs[-1]["clipiqa"],
           "clipiqa_drop": clip_recs[0]["clipiqa"] - clip_recs[-1]["clipiqa"],
           "sharpness": _nanmean([c["sharpness"] for c in clip_recs]),
           "seam_frames": [int(b) for b in trans], "d_pix_at_seams": [float(d_pix[b]) for b in trans],
           "score_sec": round(time.time() - t0, 1)}
    return rec, clip_recs


# ------------------------------------------------------------------------------------------------ discovery
def discover(roots):
    """Yield dicts describing every stitched video found under the roots (one per manifest)."""
    seen_sets = {}
    for root in roots:
        for plan_p in sorted(glob.glob(os.path.join(root, "**", "logs", "day*_plan.json"), recursive=True)):
            run_dir = os.path.dirname(os.path.dirname(plan_p))
            set_name = os.path.basename(plan_p).replace("_plan.json", "")
            if set_name in seen_sets and seen_sets[set_name] != run_dir:
                print(f"[metrics] {set_name}: ignoring second copy at {run_dir} (using {seen_sets[set_name]})")
                continue
            seen_sets[set_name] = run_dir
            plan = json.load(open(plan_p))
            fps = plan.get("config", {}).get("FPS", 24)
            for man_p in sorted(glob.glob(os.path.join(run_dir, "manifests", "*.json"))):
                man = json.load(open(man_p))
                vid = os.path.join(run_dir, "videos", f"{man['seq']}_{man['method']}_FULL.mp4")
                if not os.path.exists(vid):
                    continue
                prompts = man.get("prompts") or plan.get("prompts", {}).get(man["seq"], [])
                man.setdefault("fps", fps)
                yield {"set": set_name, "seq": man["seq"], "method": man["method"], "video": vid,
                       "manifest": man, "prompts": prompts}


def score_all(roots, out, models=None, device="cuda", clip_id=None, dino_id=None):
    os.makedirs(f"{out}/cache", exist_ok=True)
    items = list(discover(roots))
    print(f"[metrics] {len(items)} videos found:",
          {s: sum(1 for x in items if x["set"] == s) for s in sorted({x["set"] for x in items})}, flush=True)
    todo = []
    for it in items:
        key = f"{it['set']}__{it['seq']}__{it['method']}"
        it["cache"] = f"{out}/cache/{key}.json"
        size = os.path.getsize(it["video"])
        if os.path.exists(it["cache"]) and json.load(open(it["cache"])).get("video_bytes") == size:
            continue
        todo.append(it)
    if todo:
        if models is None:
            kw = {k: v for k, v in (("clip_id", clip_id), ("dino_id", dino_id)) if v}
            models = Models(device=device, **kw)
        iqa_text = models.text_clip([p for pair in IQA_PAIRS for p in pair])
        for k, it in enumerate(todo):
            try:
                rec, clips = analyse_video(it["video"], it["manifest"], it["prompts"], models, iqa_text)
                meta = {"set": it["set"], "seq": it["seq"], "method": it["method"], "video": it["video"],
                        "video_bytes": os.path.getsize(it["video"]), "lora": it["manifest"].get("lora"),
                        "anchor": it["manifest"].get("anchor"), "m": it["manifest"].get("m")}
                json.dump({**meta, "video_metrics": rec, "clips": clips}, open(it["cache"], "w"), indent=1)
                print(f"[metrics] {k + 1}/{len(todo)} {it['set']} {it['seq']} {it['method']}: "
                      f"CLIP-T {rec['clipsim']:.1f} subj {rec['subject_cons']:.3f} seam {rec['seam_pix']:.2f} "
                      f"dE {rec['deltaE_final']:.1f} over {rec['overexp_final']:.1f}% ({rec['score_sec']}s)", flush=True)
            except Exception as e:
                print(f"[metrics] FAILED {it['video']}: {e!r}", flush=True)
    return build_report(out)


# ------------------------------------------------------------------------------------------------ report
def _order(methods):
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in METHOD_ORDER)


def build_report(out):
    import pandas as pd
    recs, clips = [], []
    for p in sorted(glob.glob(f"{out}/cache/*.json")):
        d = json.load(open(p))
        base = {k: d[k] for k in ("set", "seq", "method", "lora", "anchor", "m")}
        recs.append({**base, **{k: v for k, v in d["video_metrics"].items() if not isinstance(v, list)}})
        clips += [{**base, **c} for c in d["clips"]]
    if not recs:
        print("[metrics] nothing to report"); return None
    V, Cl = pd.DataFrame(recs), pd.DataFrame(clips)
    V.to_csv(f"{out}/per_video.csv", index=False)
    Cl.to_csv(f"{out}/per_clip.csv", index=False)
    cols = [c for c in COLUMNS if c in V.columns]
    md = ["# Genesis — quantitative results", "",
          "Mean over the prompt sequences of each set (± = standard deviation across sequences). "
          "Arrows: ↑ higher is better, ↓ lower is better. Seam jump: 1.0 = seam is as smooth as a normal frame step.", ""]
    for s in sorted(V["set"].unique()):
        Vs = V[V["set"] == s]
        g = Vs.groupby("method")[cols]
        mean, std, n = g.mean(), g.std(), Vs.groupby("method").size()
        mean = mean.loc[_order(list(mean.index))]
        mean.assign(n_videos=n).to_csv(f"{out}/summary_{s}.csv")
        md += [f"## {s}  ({', '.join(sorted(Vs['seq'].unique()))})", ""]
        hdr = ["method", "n"] + [COLUMNS[c][0] + {1: " ↑", -1: " ↓", 0: ""}[COLUMNS[c][1]] for c in cols]
        md += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for mth, row in mean.iterrows():
            cells = []
            for c in cols:
                v, sd = row[c], std.loc[mth, c] if mth in std.index else float("nan")
                f = "{:.3f}" if c in ("subject_cons", "background_cons", "start_end_id", "clipiqa", "clipiqa_drop",
                                      "prompt_top1") else ("{:.0f}" if c == "sharpness" else "{:.2f}")
                cells.append("–" if pd.isna(v) else (f.format(v) + ("" if pd.isna(sd) or n[mth] < 2 else " ± " + f.format(sd))))
            md.append("| " + " | ".join([mth, str(n[mth])] + cells) + " |")
        md.append("")
        try:
            make_figures(V, Cl, s, out)
        except Exception as e:
            print("[metrics] figure error", s, repr(e))
    open(f"{out}/summary.md", "w").write("\n".join(md))
    print("\n".join(md))
    return V, Cl


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b8b7b2")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="both", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(V, Cl, s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelcolor": "#52514e", "text.color": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    os.makedirs(f"{out}/figures", exist_ok=True)
    Vs, Cs = V[V["set"] == s], Cl[Cl["set"] == s]
    methods = _order(list(Vs["method"].unique()))
    col = lambda m: METHOD_COLOR.get(m, OTHER_COLOR)
    seqs = sorted(Vs["seq"].unique())

    # 1) metric panels: one small chart per metric, methods on the y axis, bar = mean, dots = sequences
    show = [c for c in ["clipsim", "subject_cons", "background_cons", "start_end_id", "seam_pct", "deltaE_final",
                        "overexp_final", "clipiqa", "motion", "prompt_top1"] if c in Vs.columns]
    ncol = 5
    nrow = math.ceil(len(show) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 0.32 * len(methods) * nrow + 1.2 * nrow + 0.4),
                             squeeze=False)
    for ax, c in zip(axes.flat, show):
        _style(ax); ax.grid(axis="y", visible=False)
        for y, m in enumerate(methods):
            vals = Vs[Vs["method"] == m][c].dropna()
            if len(vals) == 0:
                continue
            ax.scatter(vals, [y] * len(vals), s=14, color="#8a8985", alpha=0.8, zorder=2, linewidths=0)
            ax.scatter([vals.mean()], [y], s=90, color=col(m), marker=METHOD_MARK.get(m, "o"), zorder=3,
                       edgecolor="#fcfcfb", linewidth=2)
            ax.text(1.02, y, f"{vals.mean():.3g}", va="center", fontsize=7.5, color="#0b0b0b",
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods if ax in axes[:, 0] else [""] * len(methods))
        ax.invert_yaxis()
        lab, d = COLUMNS[c]
        ax.set_title(lab + {1: "  ↑", -1: "  ↓", 0: ""}[d], loc="left")
        if c == "seam_pct":
            ax.axvline(50.0, color="#52514e", linewidth=1, linestyle=":")
        ax.margins(x=0.12, y=0.12)
    for ax in list(axes.flat)[len(show):]:
        ax.axis("off")
    fig.suptitle(f"{s}: large marker = mean over sequences (value at right), grey dots = individual sequences", x=0.01, ha="left",
                 fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(f"{out}/figures/metrics_{s}.png", dpi=160); plt.close(fig)

    # 2) drift over time: colour drift and overexposure per sequence (small multiples, one y-scale per chart)
    for c, lab in (("deltaE", "Colour drift ΔE vs first clip"), ("overexp", "Overexposed pixels (%)"),
                   ("clipiqa", "CLIP-IQA quality"), ("clipsim", "CLIP-T (prompt adherence) x100")):
        fig, axes = plt.subplots(1, len(seqs), figsize=(3.4 * len(seqs), 2.9), squeeze=False, sharey=True)
        for ax, q in zip(axes[0], seqs):
            _style(ax)
            for m in methods:
                d = Cs[(Cs["seq"] == q) & (Cs["method"] == m)].sort_values("clip")
                if len(d):
                    ax.plot(d["t_start"], d[c], color=col(m), linewidth=2, marker=METHOD_MARK.get(m, "o"),
                            markersize=5, markeredgecolor="#fcfcfb", markeredgewidth=1, label=m)
            ax.set_title(q, loc="left"); ax.set_xlabel("clip start (s)")
        axes[0][0].set_ylabel(lab)
        h, l = axes[0][0].get_legend_handles_labels()
        for ax in axes[0][1:]:
            hh, ll = ax.get_legend_handles_labels()
            for a_, b_ in zip(hh, ll):
                if b_ not in l:
                    h.append(a_); l.append(b_)
        fig.legend(h, l, loc="upper center", ncol=min(7, len(l)), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(f"{out}/figures/{c}_over_time_{s}.png", dpi=160); plt.close(fig)

    # 3) trade-off: motion vs subject consistency (method means, directly labelled)
    g = Vs.groupby("method")[["motion", "subject_cons"]].mean()
    if len(g) >= 2 and g.notna().all().all():
        fig, ax = plt.subplots(figsize=(5.2, 3.8)); _style(ax)
        for m in methods:
            if m in g.index:
                ax.scatter(g.loc[m, "motion"], g.loc[m, "subject_cons"], s=70, color=col(m),
                           marker=METHOD_MARK.get(m, "o"), edgecolor="#fcfcfb", linewidth=2, zorder=3)
                ax.annotate(m, (g.loc[m, "motion"], g.loc[m, "subject_cons"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color="#0b0b0b")
        ax.set_xlabel("Motion (optical flow, px/frame) →  more dynamic")
        ax.set_ylabel("Subject consistency (DINO) →  more stable")
        ax.set_title(f"{s}: stability vs motion", loc="left")
        ax.margins(0.2)
        fig.tight_layout(); fig.savefig(f"{out}/figures/tradeoff_{s}.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--dino", default=None)
    a = ap.parse_args()
    score_all(a.roots, a.out, device=a.device, clip_id=a.clip, dino_id=a.dino)
