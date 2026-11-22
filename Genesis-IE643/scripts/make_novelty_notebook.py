import json, os, sys

SRC = "/home/claude/g7/Genesis-IE643/src"
FILES = {n: open(f"{SRC}/{n}").read() for n in
         ["genesis_lib.py", "genesis_queue_worker.py", "genesis_metrics.py", "genesis_day6.py", "genesis_day7.py"]}

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Genesis — IE 643 | Novelty assessment: what breaks at the joins, and a targeted fix

Following the instructor's feedback from the intensive assessment:
1. **Method C + LoRA vs Method C alone** on one 20 s sequence — same prompts, same seed, same clip count (11 clips, 10 joins).
2. **Full videos with every join marked** (red border + "JOIN j" + clip number).
3. **Per-join diagnosis**: at each join — appearance jump (DINO), identity drift from the first clip, motion speed
   ratio (optical flow), colour jump. Each join is flagged with the failures it shows; the **first recurring failure**
   (earliest failure type that happens at ≥ 2 joins) is reported.
4. **Targeted fix** for that failure, run on the same sequence and compared:
   motion break → **m = 3 context (9 frames) + LoRA**;  appearance break / identity drift → **generate-and-select**
   (2 candidates per clip, keep the best). Both fixes run (one per GPU); the one matching the detected failure is the
   targeted improvement, the other is reported as a control.

384×672, 20 denoising steps (≈3 min per clip) → **about 2 hours** on Kaggle T4 x2.

**Before you run:** Input → Add Input → Your Work → Notebooks → add the **Day 3** notebook (it holds the trained LoRA).
Settings: GPU **T4 x2**, Internet **On**. Then **Save Version → Save & Run All (Commit)**; you can close the tab.
When it finishes: Output tab → download **`genesis7_novelty.zip`**.

**On Google Colab instead:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all** and keep the tab open.
Step 3 asks you to upload `lora.safetensors` and `lora_config.json` (Day 3 output → `genesis3/lora/genesis_continuation_lora/`).
With one GPU it runs C and C + LoRA, diagnoses the joins, then runs only the fix that matches the failure (~2–2.5 h).
The zip downloads automatically at the end.
""")

md("## 1 — Install")
code(r"""
!pip uninstall -y -q torchao
!pip install -q -U diffusers transformers accelerate ftfy imageio imageio-ffmpeg huggingface_hub peft pandas matplotlib psutil tabulate
!pip uninstall -y -q hf_xet
import os; os.environ["HF_HUB_DISABLE_XET"] = "1"
import diffusers, transformers, torch, peft
print("diffusers", diffusers.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__,
      "| GPUs", torch.cuda.device_count())
""")

md("## 2 — Project code (written into the output; part of the submission)")
code('import os\nROOT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"\nCODE = f"{ROOT}/genesis7/code"\nos.makedirs(CODE, exist_ok=True); print("working in", ROOT)')
for name, src in FILES.items():
    code(f"%%writefile {{CODE}}/{name}\n" + src)

md("## 3 — Configuration: LoRA, the 20 s story, the four runs")
code(r"""
import os, json, time, gc, shutil, re, html, sys, subprocess, glob, zipfile
import numpy as np
import torch

os.environ["HF_HOME"] = "/tmp/hf_home"
MODEL_ID, WEIGHTS_DIR = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "/tmp/wan22_ti2v5b"
OUT = f"{ROOT}/genesis7"
for d in ["embeds", "videos", "latents", "frames", "logs", "manifests", "code", "figures", "queue", "lora", "metrics"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
sys.path.insert(0, f"{OUT}/code")
import genesis_lib as gl
import genesis_day6 as g6
import genesis_day7 as g7
N_GPU = torch.cuda.device_count()

GEN_HOURS, SEC_PER_CLIP, POLL = 3.0, 180, 120
SECONDS = 20
HEIGHT, WIDTH, NUM_FRAMES, FPS = 384, 672, 49, 24
STEPS, GUIDANCE, FLOW_SHIFT, SEED = 20, 5.0, 3.0, 7000
CFG_CONT, ANCHOR = 3.5, 0.7
NEG_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
              "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
              "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
              "messy background, three legs, many people in the background, walking backwards")

# ---- the trained continuation LoRA (Day 3 output). Kaggle: attach the Day 3 notebook. Colab: upload the two files.
LORA_OVERRIDE = None
lora_src = []
roots = [r for r in ["/kaggle/input", "/content", "/content/drive/MyDrive"] if os.path.isdir(r)]
for top in roots:
    for root, dirs, files in os.walk(top, followlinks=True):
        depth = root[len(top):].count(os.sep)
        dirs[:] = [d for d in dirs if d not in ("drive", "sample_data", "genesis7", ".config")] if depth < 5 else []
        if "lora.safetensors" in files and "lora_config.json" in files:
            lora_src.append(root)
if LORA_OVERRIDE:
    lora_src = [LORA_OVERRIDE]
if not lora_src and ROOT == "/content":
    print("Upload lora.safetensors and lora_config.json (from the Day 3 output: genesis3/lora/genesis_continuation_lora/)")
    from google.colab import files as _cf
    up = _cf.upload()
    os.makedirs("/content/lora_upload", exist_ok=True)
    for fn, data in up.items():
        name = "lora.safetensors" if fn.endswith(".safetensors") else ("lora_config.json" if fn.endswith(".json") else fn)
        open(f"/content/lora_upload/{name}", "wb").write(data)
    if all(os.path.exists(f"/content/lora_upload/{f}") for f in ("lora.safetensors", "lora_config.json")):
        lora_src = ["/content/lora_upload"]
print("LoRA found:", lora_src)
if not lora_src:
    raise SystemExit("STOP: LoRA not found. Kaggle: attach the Day 3 notebook (Input -> Your Work -> Notebooks). "
                     "Colab: upload lora.safetensors and lora_config.json when asked.")
LORA_DIR = f"{OUT}/lora/genesis_continuation_lora"
shutil.copytree(lora_src[0], LORA_DIR, dirs_exist_ok=True)
open(f"{OUT}/queue/lora.done", "w").write(lora_src[0])

# ---- one continuous 20 s story (prompt planner: one beat per ~3.5 s, subject re-stated in every prompt)
FOX = dict(subject="red fox with a bushy white-tipped tail", style="soft golden sunrise light, steady side tracking shot.",
           template="Side view of the same {subject} {beat}, {style}",
           opening=("Side view of a red fox with a bushy white-tipped tail walking slowly from left to right through a "
                    "snowy pine forest at sunrise, the whole fox clearly visible, soft golden light, steady side tracking shot."),
           beats=["walking slowly from left to right through a snowy pine forest",
                  "walking from left to right between the pine trees, its tail held low behind it",
                  "slowing down to sniff the snow, then walking on from left to right",
                  "walking from left to right past a fallen log covered in snow",
                  "walking from left to right up a gentle snowy slope",
                  "reaching the top of the slope as sunlight shines through the trees"])

def story(m):
    its, info = g6.plan_script(FOX["subject"], FOX["style"], FOX["beats"], SECONDS, FPS, NUM_FRAMES, m,
                               opening=FOX["opening"], template=FOX["template"], key_prefix="fox")
    for i, it in enumerate(its):
        it["seed"] = SEED + i                 # the same seed for clip i in every run
    print(f"m={m}: {info}")
    return its

ITEMS2, ITEMS3 = story(2), story(3)
PROMPTS = {**g6.prompts_needed(ITEMS2), **g6.prompts_needed(ITEMS3)}

C_ARGS = dict(cfg_cont=CFG_CONT, anchor=ANCHOR, finalize=True)
JOBS = [
    {"id": "first", "name": "fox20", "items": ITEMS2[:1], "method": "text", "lora": False, "finalize": False, "deps": [], "m": 2},
    {"id": "C", "name": "fox20", "items": ITEMS2, "method": "C_anchor", "lora": False, "deps": ["first"], "m": 2, **C_ARGS},
    {"id": "C_lora", "name": "fox20", "items": ITEMS2, "method": "C_anchor_lora", "lora": True, "deps": ["first", "lora"], "m": 2, **C_ARGS},
    {"id": "fix_m3", "name": "fox20", "items": ITEMS3, "method": "C_anchor_lora_m3", "lora": True, "deps": ["first", "lora"], "m": 3, **C_ARGS},
    {"id": "fix_select", "name": "fox20", "items": ITEMS2, "method": "C_anchor_lora_select", "lora": True, "deps": ["first", "lora"],
     "m": 2, "n_cand": 2, **C_ARGS},
]
LABEL = {"C_anchor": "C", "C_anchor_lora": "C + LoRA", "C_anchor_lora_m3": "C + LoRA, m=3 (fix)",
         "C_anchor_lora_select": "C + LoRA, select (fix)"}
CONFIG = {"OUT": OUT, "WEIGHTS_DIR": WEIGHTS_DIR, "EMBEDS": f"{OUT}/embeds/day7_embeds.pt", "HEIGHT": HEIGHT,
          "WIDTH": WIDTH, "NUM_FRAMES": NUM_FRAMES, "STEPS": STEPS, "GUIDANCE": GUIDANCE, "FLOW_SHIFT": FLOW_SHIFT,
          "FPS": FPS, "DEVICE": "cuda", "VAE_OFFLOAD": True, "DTYPE": "float16", "LORA_DIR": LORA_DIR,
          "SAVE_LATENTS": False, "DEADLINE": None, "SEC_PER_CLIP": SEC_PER_CLIP, "SCORER_MODELS": {}}
PLAN = f"{OUT}/logs/day7_plan.json"
json.dump({"config": CONFIG, "jobs": JOBS, "negative_prompt": NEG_PROMPT, "story": FOX}, open(PLAN, "w"), indent=2)
print([j["id"] for j in JOBS])
""")

md("## 4 — Text encoder: encode the prompts, then free the memory")
code(r"""
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy
snapshot_download(MODEL_ID, local_dir=WEIGHTS_DIR, allow_patterns=["model_index.json", "tokenizer/*", "text_encoder/*"])
tok = AutoTokenizer.from_pretrained(f"{WEIGHTS_DIR}/tokenizer")
if N_GPU >= 2:
    te = UMT5EncoderModel.from_pretrained(f"{WEIGHTS_DIR}/text_encoder", torch_dtype=torch.float32,
                                          device_map="auto", max_memory={0: "14GiB", 1: "14GiB", "cpu": "20GiB"})
    te_device = "cuda:0"
elif N_GPU == 1:   # Colab / single T4: bf16 straight onto the GPU (fits in 16 GB, little CPU RAM needed)
    te = UMT5EncoderModel.from_pretrained(f"{WEIGHTS_DIR}/text_encoder", torch_dtype=torch.bfloat16, device_map={"": 0})
    te_device = "cuda:0"
else:
    te = UMT5EncoderModel.from_pretrained(f"{WEIGHTS_DIR}/text_encoder", torch_dtype=torch.bfloat16); te_device = "cpu"
te.eval()

def prompt_clean(t):
    t = ftfy.fix_text(t); t = html.unescape(html.unescape(t)).strip()
    return re.sub(r"\s+", " ", t).strip()

@torch.no_grad()
def t5_embed(text, max_len=512):
    enc = tok([prompt_clean(text)], padding="max_length", max_length=max_len, truncation=True,
              add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    ids, mask = enc.input_ids.to(te_device), enc.attention_mask.to(te_device)
    n = int(mask.sum())
    h = te(ids, mask).last_hidden_state[0, :n].float().cpu()
    out = torch.zeros(1, max_len, h.shape[-1]); out[0, :n] = h
    return out.half()

E = {"__negative__": t5_embed(NEG_PROMPT), "__empty__": t5_embed("")}
for k, p in PROMPTS.items():
    E[k] = t5_embed(p)
torch.save(E, CONFIG["EMBEDS"]); print("prompts encoded:", len(E))
del te; gc.collect(); torch.cuda.empty_cache()
shutil.rmtree(f"{WEIGHTS_DIR}/text_encoder", ignore_errors=True); shutil.rmtree("/tmp/hf_home", ignore_errors=True)
""")

md("## 5 — Download transformer, VAE and scheduler")
code(r"""
snapshot_download(MODEL_ID, local_dir=WEIGHTS_DIR, allow_patterns=["model_index.json", "transformer/*", "vae/*", "scheduler/*"])
""")

md("## 6 — Generate (one worker per GPU: C ‖ C+LoRA, then m=3 fix ‖ select fix)")
code(r"""
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
P = json.load(open(PLAN)); P["config"]["DEADLINE"] = time.time() + GEN_HOURS * 3600
json.dump(P, open(PLAN, "w"), indent=2)
for f in os.listdir(f"{OUT}/queue"):
    if f.endswith(".claimed") and not os.path.exists(f"{OUT}/queue/{f[:-8]}.done"):
        os.remove(f"{OUT}/queue/{f}")

def launch(args, gpu, tag):
    logf = open(f"{OUT}/logs/{tag}.log", "a")
    return subprocess.Popen([sys.executable, *args], env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdout=logf, stderr=subprocess.STDOUT)
def last_line(tag):
    p = f"{OUT}/logs/{tag}.log"
    lines = open(p).read().strip().splitlines() if os.path.exists(p) else []
    return lines[-1][:140] if lines else "-"

def run_workers(jobs, tag):
    P = json.load(open(PLAN)); P["jobs"] = jobs; json.dump(P, open(f"{OUT}/logs/day7_plan_{tag}.json", "w"), indent=2)
    procs = {f"worker{w}": launch([f"{OUT}/code/genesis_queue_worker.py", "--plan", f"{OUT}/logs/day7_plan_{tag}.json",
                                    "--worker", str(w)], w, f"worker{w}") for w in range(max(1, min(N_GPU, 2)))}
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(POLL)
        print(f"[{tag} {(time.time() - t0) / 60:5.0f} min] " + " | ".join(f"{k}: {last_line(k)}" for k in procs), flush=True)

if N_GPU >= 2:
    run_workers(JOBS, "all")
else:
    # one GPU: the fix is chosen from the diagnosis of C + LoRA before it is run (the exact protocol, sequentially)
    run_workers([j for j in JOBS if j["id"] in ("first", "C", "C_lora")], "phase1")
    import genesis_metrics as gm
    _M = gm.Models(device="cuda" if torch.cuda.is_available() else "cpu")
    _man = json.load(open(f"{OUT}/manifests/fox20_C_anchor_lora.json")); _man["fps"] = FPS
    _rows, _ = g7.join_diagnosis(gm.read_u8(f"{OUT}/videos/fox20_C_anchor_lora_FULL.mp4"), _man, _M)
    _f, _j, _n = g7.first_recurring_failure(_rows)
    _fix = {"m3": "fix_m3", "select": "fix_select"}.get(g7.FIX_FOR.get(_f), "fix_m3")
    print(f"first recurring failure: {_f} (join {_j}, {_n} joins) -> running {_fix}")
    del _M; gc.collect(); torch.cuda.empty_cache()
    run_workers([j for j in JOBS if j["id"] == _fix], "phase2")
STATE = {k: sorted(f[:-len(k)] for f in os.listdir(f'{OUT}/queue') if f.endswith(k)) for k in ['.done', '.failed', '.skipped']}
print(STATE)
for f in os.listdir(f"{OUT}/queue"):
    if f.endswith(".failed"):
        print("FAILED:", f, open(f"{OUT}/queue/{f}").read()[-1500:])
""")

md(r"""
## 7 — Diagnose every join, find the first recurring failure, compare the fix
""")
code(r"""
from IPython.display import Markdown, display
import pandas as pd
import genesis_metrics as gm
r = subprocess.run([sys.executable, f"{OUT}/code/genesis_metrics.py", "--roots", OUT, "--out", f"{OUT}/metrics"],
                   env=dict(env, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
open(f"{OUT}/logs/metrics.log", "w").write(r.stdout + r.stderr); print((r.stdout + r.stderr)[-1500:])

M = gm.Models(device="cuda" if torch.cuda.is_available() else "cpu")
ORDER = ["C_anchor", "C_anchor_lora", "C_anchor_lora_m3", "C_anchor_lora_select"]
VID, MAN, ROWS, SPEED = {}, {}, {}, {}
for k in ORDER:
    mp, vp = f"{OUT}/manifests/fox20_{k}.json", f"{OUT}/videos/fox20_{k}_FULL.mp4"
    if not (os.path.exists(mp) and os.path.exists(vp)):
        print("missing:", k); continue
    MAN[k], VID[k] = json.load(open(mp)), gm.read_u8(vp)
    MAN[k]["fps"] = FPS
    ROWS[k], SPEED[k] = g7.join_diagnosis(VID[k], MAN[k], M)
    pd.DataFrame(ROWS[k]).to_csv(f"{OUT}/metrics/joins_{k}.csv", index=False)

# per-join table for the two main runs
for k in ["C_anchor", "C_anchor_lora"]:
    if k in ROWS:
        d = pd.DataFrame(ROWS[k])
        display(Markdown(f"### Joins — {LABEL[k]}\n" + d.round(3).to_markdown(index=False)))

# first recurring failure, and the fix it calls for
FINDINGS = {}
for k in ROWS:
    f, j, n = g7.first_recurring_failure(ROWS[k])
    FINDINGS[k] = {"first_recurring_failure": f, "first_join": j, "joins_with_it": n, "of_joins": len(ROWS[k])}
base = "C_anchor_lora" if "C_anchor_lora" in FINDINGS else "C_anchor"
F0 = FINDINGS.get(base, {}).get("first_recurring_failure")
FIX = {"m3": "C_anchor_lora_m3", "select": "C_anchor_lora_select"}.get(g7.FIX_FOR.get(F0)) if F0 else None
FINDINGS["_decision"] = {"diagnosed_on": base, "failure": F0, "targeted_fix": FIX}
json.dump(FINDINGS, open(f"{OUT}/metrics/findings.json", "w"), indent=2)
print(json.dumps(FINDINGS, indent=1))

# one table: standard metrics + join summary for every run
cache = {c["method"]: c["video_metrics"] for c in (json.load(open(p)) for p in glob.glob(f"{OUT}/metrics/cache/*.json"))}
T = []
for k in ROWS:
    vm = cache.get(k, {})
    T.append({"run": LABEL[k], "clips": len(MAN[k]["clip_lengths"]), "seconds": MAN[k].get("seconds"),
              "start-end identity ↑": vm.get("start_end_id"), "subject cons. ↑": vm.get("subject_cons"),
              "seam pct ↓": vm.get("seam_pct"), "ΔE ↓": vm.get("deltaE_final"), "CLIP-T ↑": vm.get("clipsim"),
              "prompt progress ↑": vm.get("prompt_progress"), "motion": vm.get("motion"),
              **{k2: v for k2, v in g7.summary(ROWS[k]).items()}})
T = pd.DataFrame(T)
T.to_csv(f"{OUT}/metrics/novelty_table.csv", index=False)
open(f"{OUT}/metrics/novelty_table.md", "w").write(T.round(3).to_markdown(index=False))
display(Markdown("### All runs\n" + T.round(3).to_markdown(index=False)))
if F0:
    display(Markdown(f"**First recurring failure ({LABEL[base]}): {F0}** — first at join {FINDINGS[base]['first_join']}, "
                     f"at {FINDINGS[base]['joins_with_it']}/{FINDINGS[base]['of_joins']} joins → targeted fix: **{LABEL.get(FIX, 'none')}**"))
    if FIX in ROWS:
        display(Markdown(f"After the fix: {F0} at {sum(r[F0] for r in ROWS[FIX])}/{len(ROWS[FIX])} joins "
                         f"(was {FINDINGS[base]['joins_with_it']}/{FINDINGS[base]['of_joins']})."))

# generate-and-select log
recs = []
for f in glob.glob(f"{OUT}/logs/runs_worker*.jsonl"):
    recs += [json.loads(l) for l in open(f) if l.strip()]
sel = [x for x in recs if "chosen" in x]
if sel:
    S = pd.DataFrame([{"clip": x["clip"], "chosen": x["chosen"], **{f"{m}_{c}": x["scores"][c][m] for c in range(2)
                       for m in ("identity", "clip_t", "seam", "total")}} for x in sel]).sort_values("clip")
    S.to_csv(f"{OUT}/metrics/select_log.csv", index=False)
    print(f"select: 2nd candidate kept in {int((S.chosen == 1).sum())}/{len(S)} clips")
""")

md("## 8 — Figures and join-marked videos")
code(r"""
from IPython.display import Image as IPImage
from PIL import Image, ImageDraw
from diffusers.utils import export_to_video
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"C_anchor": "#8a8985", "C_anchor_lora": "#2a6fdb", "C_anchor_lora_m3": "#d9572b", "C_anchor_lora_select": "#1f9e6e"}

def lab(img, text):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)
def small(f, s=2):
    return np.asarray(Image.fromarray(f).resize((WIDTH // s, HEIGHT // s)))

# (a) per-join curves
fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
spec = [("appearance_jump", "appearance jump at join (1 = normal)", g7.THRESH["appearance"]),
        ("identity_to_start", "identity vs first clip (DINO)", g7.THRESH["identity"]),
        ("speed_ratio", "speed after / before join", None),
        ("colour_jump", "colour jump at join (1 = normal)", g7.THRESH["colour"])]
for a, (key, title, th) in zip(axs, spec):
    for k in ROWS:
        d = pd.DataFrame(ROWS[k]); a.plot(d["time_s"], d[key], "o-", ms=4, lw=1.5, color=COL[k], label=LABEL[k])
    if th: a.axhline(th, ls="--", color="#c33", lw=1)
    if key == "speed_ratio":
        for th2 in (g7.THRESH["speed_lo"], g7.THRESH["speed_hi"]): a.axhline(th2, ls="--", color="#c33", lw=1)
        a.axhline(1, color="#999", lw=0.8)
    a.set_title(title, fontsize=10); a.set_xlabel("join time (s)"); a.grid(alpha=0.3)
axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/joins_curves.png", dpi=160); plt.close(fig)
display(IPImage(f"{OUT}/figures/joins_curves.png", width=1200))

# (b) motion speed along the video, joins dotted
fig, ax = plt.subplots(figsize=(12, 3.2))
for k in ROWS:
    sp = pd.Series(SPEED[k]).rolling(5, center=True, min_periods=1).mean()
    ax.plot(np.arange(len(sp)) / FPS, sp, lw=1.4, color=COL[k], label=LABEL[k])
for b in MAN.get("C_anchor_lora", MAN[next(iter(MAN))])["boundaries"]:
    ax.axvline(b / FPS, color="#bbb", lw=0.7, ls=":")
ax.set_xlabel("time (s) — dotted = joins (m=2)"); ax.set_ylabel("motion (px/frame)"); ax.legend(fontsize=8, ncol=4)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/speed_timeline.png", dpi=160); plt.close(fig)
display(IPImage(f"{OUT}/figures/speed_timeline.png", width=1200))

# (c) frames just before / after every join (C vs C+LoRA, and the fix)
rows_img = []
for k in [x for x in ["C_anchor", "C_anchor_lora", FIX] if x in VID]:
    v, b = VID[k], MAN[k]["boundaries"]
    for side, off in (("before", -1), ("after", 0)):
        tiles = [lab(small(v[s + off], 4), f"{LABEL[k][:14]} J{j + 1} {side}") for j, s in enumerate(b)]
        rows_img.append(np.concatenate(tiles, 1))
wmax = max(r.shape[1] for r in rows_img)
rows_img = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows_img]
Image.fromarray(np.concatenate(rows_img, 0)).save(f"{OUT}/figures/join_frames.jpg", quality=90)
display(IPImage(f"{OUT}/figures/join_frames.jpg", width=1400))

# (d) timeline every 4 s
rows_img = []
for k in ROWS:
    v = VID[k]; ts = [min(int(s * FPS), len(v) - 1) for s in (0, 4, 8, 12, 16, 20)]
    rows_img.append(np.concatenate([lab(small(v[t], 3), f"{LABEL[k]} {t / FPS:.0f}s") for t in ts], 1))
Image.fromarray(np.concatenate(rows_img, 0)).save(f"{OUT}/figures/timeline.jpg", quality=90)
display(IPImage(f"{OUT}/figures/timeline.jpg", width=1200))

# (e) join-marked videos: each run, and a 2x2 grid of all runs (demo video)
MK = {k: g7.mark_joins(VID[k], MAN[k]["boundaries"], LABEL[k], fps=FPS) for k in VID}
for k, v in MK.items():
    export_to_video(list(v.astype(np.float32) / 255), f"{OUT}/videos/fox20_{k}_JOINS.mp4", fps=FPS)
L = min(len(v) for v in MK.values())
tiles = [np.stack([small(f) for f in MK[k][:L]]) for k in ORDER if k in MK]
while len(tiles) < 4:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.concatenate([np.concatenate(tiles[:2], 2), np.concatenate(tiles[2:4], 2)], 1)
export_to_video(list(grid.astype(np.float32) / 255), f"{OUT}/videos/fox20_ALL_RUNS_JOINS_MARKED.mp4", fps=FPS)
print("videos written")

# zip
for d in ["frames", "embeds", "latents", "lora"]:
    shutil.rmtree(f"{OUT}/{d}", ignore_errors=True)
keep = lambda r: (not r.startswith("genesis7/videos/") or r.endswith(("_FULL.mp4", "_JOINS.mp4", "_MARKED.mp4"))) \
                 and "/metrics/cache/" not in r
with zipfile.ZipFile(f"{ROOT}/genesis7_novelty.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in files:
            p = os.path.join(root, fn); rel = os.path.relpath(p, os.path.dirname(OUT))
            if keep(rel):
                z.write(p, rel)
print("zip:", round(os.path.getsize(f"{ROOT}/genesis7_novelty.zip") / 2**20, 1), "MB")
import psutil
for ch in psutil.Process().children(recursive=True):
    try: ch.kill()
    except Exception: pass
print("ALL DONE")
if ROOT == "/content":
    from google.colab import files as _cf
    _cf.download(f"{ROOT}/genesis7_novelty.zip")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/Genesis_IE643_Novelty_Joins_and_Fix.ipynb"
json.dump(nb, open(path, "w"), indent=1)
print(path, len(cells), "cells")
