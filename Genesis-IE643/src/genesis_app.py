"""genesis_app.py — Gradio interface for Genesis (IE 643): long videos from chained short Wan2.2 TI2V-5B clips.

Tab 1 "Generate": prompts (one per line, a line "[CUT]" starts a new shot) + target duration + chaining method +
         speed preset -> stitched video, per-clip strip, metric table and per-clip plots (computed with
         genesis_metrics, the same code used for the report).
Tab 2 "Results gallery": the videos generated on Days 3-5 with their metrics.
Tab 3 "Overall metrics": the report tables and figures.
Tab 4 "How it works".
"""
import glob, json, os, threading, time, traceback

import numpy as np
import torch

import genesis_lib as gl
import genesis_metrics as gm

METHODS = {   # label: (method name, m, cfg_cont, anchor, lora)
    "C + LoRA (final method)": ("C_anchor_lora", 2, 3.5, 0.7, True),
    "C: B + colour anchoring": ("C_anchor", 2, 3.5, 0.7, False),
    "B + LoRA": ("B_m2_lora", 2, None, 0.0, True),
    "B: last 5 frames (m=2)": ("B_m2", 2, None, 0.0, False),
    "A: last frame (m=1)": ("A_lastframe", 1, None, 0.0, False),
    "Baseline: independent clips, hard cuts": ("baseline", 1, None, 0.0, False),
    "C + LoRA, m=3 (experimental: 9-frame context)": ("C_anchor_lora_m3", 3, 3.5, 0.7, True),
}
PRESETS = {   # label: (height, width, frames, steps, est. seconds per clip on a T4)
    "Live demo (320x576, 33 frames, 16 steps)": (320, 576, 33, 16, 70),
    "Fast preview (384x672, 20 steps)": (384, 672, 49, 20, 170),
    "Report quality (480x832, 30 steps)": (480, 832, 49, 30, 420),
}
EXAMPLE = """Side view of a red fox walking slowly from left to right through a snowy pine forest at sunrise, soft golden light, steady side tracking shot.
Side view of the same red fox slowing down to sniff the snow, then walking on from left to right, soft golden light, steady side tracking shot.
[CUT]
Close-up of a red fox's face in a snowy forest at sunrise, its ears twitching as it listens, soft golden light, shallow depth of field."""

METRIC_ROWS = [   # (key, label, how to read it)
    ("clipsim", "Prompt adherence (CLIP-T x100)", "higher = frames match their clip's prompt"),
    ("prompt_progress", "Prompt progress", "> 0 = the video moves on to later prompts"),
    ("subject_cons", "Subject consistency (DINO)", "1.0 = the subject looks the same in every frame"),
    ("background_cons", "Background consistency (CLIP)", "1.0 = the scene stays the same"),
    ("start_end_id", "Start-end identity (DINO)", "first vs last clip of each shot (1.0 = same look)"),
    ("seam_pix", "Seam jump (pixel)", "1.0 = a seam is as smooth as an ordinary frame step"),
    ("seam_pct", "Seam percentile", "~50 = seams look like ordinary frame steps, 100 = hard cut"),
    ("cut_pix", "Intentional [CUT] jump", "large by design"),
    ("deltaE_final", "Colour drift dE (last clip)", "lower = colour stays like the start of the shot"),
    ("overexp_final", "Overexposed pixels % (last clip)", "lower = no blown-out highlights"),
    ("clipiqa", "Quality (CLIP-IQA, 0-1)", "higher = sharper, better exposed"),
    ("motion", "Motion (px/frame)", "how much moves; very low = static"),
]


def plan_clips(prompts_text, seconds, method, fps, nf, seed):
    """Text box + duration -> list of clip items for genesis_lib.run_script, and a note for the user."""
    name, m = method[0], method[1]
    lines = [l for l in prompts_text.splitlines() if l.strip()]
    items = gl.parse_script(lines)
    if not items:
        raise ValueError("Enter at least one prompt.")
    if name == "baseline":
        n = max(1, int(np.ceil(seconds * fps / nf)))
    else:
        n = gl.clips_for_duration(seconds, fps, nf, m)
    note = ""
    if n < len(items):
        note = f"{len(items)} prompts need at least {len(items)} clips, so the video is longer than {seconds:g} s."
        n = len(items)
    per_prompt = n / len(items)
    if per_prompt > 2.5:
        sec_per_clip = (nf - (gl.context_frames(m) if name != "baseline" else 0)) / fps
        note += (f"{' ' if note else ''}⚠ Each prompt is used for ~{per_prompt:.0f} clips (~{per_prompt * sec_per_clip:.0f} s), "
                 "so the video will probably stall. Add more prompts (about one per 4 s) or shorten the duration.")
    clips = gl.expand_script(items, n)
    for i, c in enumerate(clips):
        c.update(seed=int(seed) + i)
        if name == "baseline":
            c["cut"] = i > 0          # baseline: every clip is generated from text -> hard cut at every seam
    return clips, note


class App:
    def __init__(self, G, encode, metric_models, lora_dir, work_dir, gallery_roots=(), presets=None, fps=24):
        self.G, self.encode, self.M, self.lora_dir = G, encode, metric_models, lora_dir
        self.work, self.fps = work_dir, fps
        self.presets = presets or PRESETS
        self.gallery_roots = list(gallery_roots)
        self.lock = threading.Lock()
        os.makedirs(work_dir, exist_ok=True)

    # ------------------------------------------------------------------ generation (a generator: streams progress)
    def generate(self, prompts_text, seconds, method_label, preset_label, seed):
        method = METHODS[method_label]
        name, m, cfg_cont, anchor, lora = method
        h, w, nf, steps, est = self.presets[preset_label]
        if lora and not (self.lora_dir and os.path.exists(f"{self.lora_dir}/lora.safetensors")):
            yield None, "**LoRA not found** — attach the Day 3 notebook output, or pick a method without LoRA.", None, None, None
            return
        try:
            clips, note = plan_clips(prompts_text, float(seconds), method, self.fps, nf, seed)
        except ValueError as e:
            yield None, f"**{e}**", None, None, None
            return
        if not self.lock.acquire(blocking=False):
            yield None, "Another video is being generated — please wait for it to finish.", None, None, None
            return
        try:
            run = time.strftime("run_%H%M%S")
            out = f"{self.work}/{run}"
            for d in ("videos", "frames", "latents", "manifests", "logs"):
                os.makedirs(f"{out}/{d}", exist_ok=True)
            n = len(clips)
            yield None, f"Encoding {len(set(c['prompt'] for c in clips))} prompt(s)…", None, None, None
            E = {"__negative__": self.encode(None)}
            cache = {}
            for i, c in enumerate(clips):
                if c["prompt"] not in cache:
                    cache[c["prompt"]] = self.encode(c["prompt"])
                c["key"] = f"p{i}"
                E[c["key"]] = cache[c["prompt"]]
            G = self.G
            G.h, G.w, G.nf, G.steps = h, w, nf, steps
            gl.SAVE_LATENTS = False
            log = f"{out}/logs/run.jsonl"
            state = {"err": None}

            def work():
                try:
                    gl.run_script(G, E, out, "video", clips, name, m, log, cfg_cont=cfg_cont, anchor=anchor,
                                  lora=self.lora_dir if lora else None)
                except Exception:
                    state["err"] = traceback.format_exc()

            th = threading.Thread(target=work, daemon=True)
            th.start()
            t0 = time.time()
            header = (f"**{method_label}** · {n} clips · {preset_label}" + (f"  \n{note}" if note else "") +
                      f"  \nRough estimate ≈ {n * est / 60:.0f} min on a T4 (clips are made one after another).")
            while th.is_alive():
                done = sum(1 for _ in open(log)) if os.path.exists(log) else 0
                el = time.time() - t0
                eta = (el / done * (n - done)) if done else (n * est - el)
                yield None, f"{header}  \nGenerating clip {min(done + 1, n)}/{n} · {el / 60:.1f} min elapsed · ~{max(eta, 0) / 60:.0f} min left", None, None, None
                th.join(timeout=5)
            if state["err"]:
                yield None, f"**Generation failed**\n```\n{state['err'][-1500:]}\n```", None, None, None
                return
            video = f"{out}/videos/video_{name}_FULL.mp4"
            man = json.load(open(f"{out}/manifests/video_{name}.json"))
            yield video, f"{header}  \nDone in {(time.time() - t0) / 60:.1f} min ({man['seconds']} s of video). Computing metrics…", None, None, None
            rec, per_clip = gm.analyse_video(video, dict(man, fps=self.fps), man["prompts"], self.M)
            table = metric_table(rec)
            yield (video, f"{header}  \nDone: {man['seconds']} s, {n} clips, {len(man.get('cuts', []))} cut(s), "
                          f"generated in {(time.time() - t0) / 60:.1f} min.",
                   table, clip_plot(per_clip, self.fps), strip(video, man, self.fps))
        finally:
            self.lock.release()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ gallery
    def gallery_items(self):
        items = {}
        for root in self.gallery_roots:
            for v in sorted(glob.glob(f"{root}/**/videos/*_FULL.mp4", recursive=True) +
                            glob.glob(f"{root}/**/videos/*_COMPARE.mp4", recursive=True)):
                run = os.path.basename(os.path.dirname(os.path.dirname(v)))
                items[f"{run} · {os.path.basename(v)[:-4]}"] = v
        return items

    def gallery_show(self, key):
        items = self.gallery_items()
        if key not in items:
            return None, None
        v = items[key]
        base = os.path.basename(v)[:-4]
        if base.endswith("_COMPARE"):
            return v, None
        run_dir = os.path.dirname(os.path.dirname(v))
        seq_method = base[:-5]
        for root in self.gallery_roots:
            for p in glob.glob(f"{root}/**/metrics/per_video.csv", recursive=True):
                import pandas as pd
                V = pd.read_csv(p)
                V["key"] = V["seq"] + "_" + V["method"]
                hit = V[V["key"] == seq_method]
                if len(hit):
                    return v, metric_table(hit.iloc[-1].to_dict())
        return v, None


def metric_table(rec):
    import pandas as pd
    rows = []
    for k, label, how in METRIC_ROWS:
        v = rec.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        rows.append({"metric": label, "value": round(float(v), 3), "how to read it": how})
    return pd.DataFrame(rows)


def clip_plot(per_clip, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [c["t_start"] for c in per_clip]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, key, title, color in ((axes[0], "clipsim", "Prompt adherence (CLIP-T x100)", "#2a78d6"),
                                  (axes[1], "deltaE", "Colour drift dE vs shot start", "#eb6834"),
                                  (axes[2], "motion", "Motion (px/frame)", "#1baf7a")):
        ax.plot(t, [c[key] for c in per_clip], color=color, linewidth=2, marker="o", markersize=5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("clip start (s)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#ecebe7")
    fig.tight_layout()
    return fig


def strip(video, man, fps, width=208):
    """First frame of every clip, side by side (cuts are marked with a red bar)."""
    import cv2
    fr = gm.read_u8(video)
    starts = [0] + list(man["boundaries"])
    cuts = set(man.get("cuts", []))
    tiles = []
    for s in starts:
        f = fr[min(s, len(fr) - 1)]
        h = int(round(f.shape[0] * width / f.shape[1]))
        t = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
        if s in cuts:
            t[:, :4] = (230, 60, 60)
        tiles.append(t)
    return np.concatenate(tiles, axis=1)


HOW_IT_WORKS = r"""
### Pipeline
1. Your prompts are split into **shots** at every `[CUT]` line, then spread over the clips needed for the duration
   (each clip is 49 frames ≈ 2 s at 24 fps; a chained clip adds 44 new frames with m = 2).
2. The **first clip of each shot** is generated from text only (Wan2.2 TI2V-5B, flow matching, 30 or 20 UniPC steps).
3. Every **following clip** is conditioned on the last *K = 1 + 4(m−1)* frames of the previous clip: they are encoded
   by the VAE into *m* clean latent frames, given timestep 0 and re-injected after every solver step, so the model
   only generates the rest of the clip. The repeated *K* frames are dropped when stitching.
4. **Method C** pulls each new frame's LAB colour mean/std toward the first clip of the shot (strength 0.7) and uses a
   lower CFG (3.5) on continuation clips, which stops exposure/colour drift.
5. **LoRA**: a rank-32 adapter on the attention projections, trained for 800 steps on 300 OpenVid-1M clips to continue
   real videos from m = 1–3 clean context frames (loss only on the frames to be generated).
6. The finished video is scored with the same metrics as the report (CLIP, DINO, optical flow, LAB colour).

### Known limitations (measured)
- Chained takes **stall**: they follow later prompts less than independent clips do.
- A 5-frame context carries appearance but little velocity: motion **slows down at many seams**. The m = 3 option
  tests whether a 9-frame context helps.
- Colour anchoring makes seams slightly less smooth than plain B.
"""


def build(app, overall_md=None, overall_figs=()):
    import gradio as gr
    with gr.Blocks(title="Genesis — long video from short clips") as demo:
        gr.Markdown("# Genesis — long-form video by chaining short diffusion clips\n"
                    "IE 643 · Team Genesis (23b1311) · Wan2.2 TI2V-5B + continuation LoRA")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompts = gr.Textbox(label="Prompts — one per line; a line [CUT] starts a new shot", lines=9,
                                         value=EXAMPLE)
                with gr.Column(scale=2):
                    seconds = gr.Slider(2, 30, value=6, step=1, label="Target duration (seconds)")
                    method = gr.Dropdown(list(METHODS), value=list(METHODS)[0], label="Chaining method")
                    preset = gr.Radio(list(app.presets), value=list(app.presets)[0], label="Speed / quality")
                    seed = gr.Number(value=7000, precision=0, label="Seed")
                    go = gr.Button("Generate", variant="primary")
            status = gr.Markdown()
            video = gr.Video(label="Stitched video", autoplay=True)
            strip_img = gr.Image(label="First frame of every clip (red bar = [CUT])", type="numpy")
            with gr.Row():
                table = gr.Dataframe(label="Metrics for this video", wrap=True,
                                     headers=["metric", "value", "how to read it"])
            plot = gr.Plot(label="Per-clip metrics")
            go.click(app.generate, [prompts, seconds, method, preset, seed], [video, status, table, plot, strip_img], show_progress="minimal")
        with gr.Tab("Results gallery"):
            keys = list(app.gallery_items())
            pick = gr.Dropdown(keys, value=keys[0] if keys else None, label="Generated on Days 3–5")
            gv = gr.Video(label="Video")
            gt = gr.Dataframe(label="Metrics", wrap=True, headers=["metric", "value", "how to read it"])
            pick.change(app.gallery_show, pick, [gv, gt])
            demo.load(app.gallery_show, pick, [gv, gt])
        with gr.Tab("Overall metrics"):
            gr.Markdown(overall_md or "No metric summary found (attach the Day 5 output).")
            for f in overall_figs:
                gr.Image(f, label=os.path.basename(f), type="filepath")
        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS)
    return demo
