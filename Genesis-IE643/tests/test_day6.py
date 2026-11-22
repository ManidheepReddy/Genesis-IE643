"""End-to-end CPU test of the Day 6 notebook: runs its real cells with a tiny random Wan2.2-shaped model, tiny CLIP/DINO,
the tiny Day 3 LoRA and a fake OpenVid archive (CSV + zip of small mp4s) for the held-out step."""
import csv, glob, json, os, shutil, subprocess, sys, time, types, zipfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
NB = f"{HERE}/t_nb.ipynb"
subprocess.run([sys.executable, f"{HERE}/make_day6.py", NB], check=True)
cells = json.load(open(NB))["cells"]
TM = f"{SP}/day4/t_models"

# ---- fake OpenVid: 30 small videos in a zip + caption CSV
FK = "/tmp/t6fake_data"
shutil.rmtree(FK, ignore_errors=True); os.makedirs(f"{FK}/vids")
import imageio.v3 as iio
rng = np.random.default_rng(0)
rows = []
for i in range(200):
    base = rng.integers(0, 255, (1, 36, 48, 3))
    fr = np.stack([np.roll(base[0], 2 * t, axis=1) for t in range(40)]).astype(np.uint8)
    iio.imwrite(f"{FK}/vids/v{i:03d}.mp4", fr, fps=24)
    rows.append({"video": f"v{i:03d}.mp4", "caption": f"a moving pattern number {i}", "aesthetic score": 5 + (i * 7919) % 13,
                 "motion score": 1 + i % 5, "temporal consistency score": 0.9 + (i % 3) / 100, "seconds": 2.0 + i % 3})
with zipfile.ZipFile(f"{FK}/part0.zip", "w") as z:
    for r in rows:
        z.write(f"{FK}/vids/{r['video']}", f"OpenVid_part0/{r['video']}")
with open(f"{FK}/ov.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fake = types.ModuleType("t6fake")
fake.hf_hub_download = lambda repo, fn, **k: f"{FK}/ov.csv" if fn.endswith(".csv") else f"{FK}/part0.zip"
class _FS:
    def open(self, *a, **k): return open(f"{FK}/part0.zip", "rb")
fake.HfFileSystem = _FS
sys.modules["t6fake"] = fake
# lpips needs AlexNet weights from the internet: stand-in with the same interface
lp = types.ModuleType("lpips")
class _L(torch.nn.Module):
    def __init__(self, **k): super().__init__()
    def forward(self, a, b): return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)
lp.LPIPS = _L
sys.modules["lpips"] = lp

shutil.rmtree("/kaggle", ignore_errors=True)
os.makedirs("/kaggle/working"); os.makedirs("/kaggle/input/genesis-ie643-day3/genesis3")
shutil.copytree(f"{SP}/day3/tiny_out3/lora", "/kaggle/input/genesis-ie643-day3/genesis3/lora")

SUBS = [
    ('HEIGHT, WIDTH, NUM_FRAMES, FPS = 384, 672, 49, 24', 'HEIGHT, WIDTH, NUM_FRAMES, FPS = 64, 64, 13, 8'),
    ('STEPS, GUIDANCE, FLOW_SHIFT, BASE_SEED = 25, 5.0, 3.0, 6000', 'STEPS, GUIDANCE, FLOW_SHIFT, BASE_SEED = 2, 5.0, 3.0, 6000'),
    ('story(FOX, 30,', 'story(FOX, 4,'), ('story(TRAIN, 60,', 'story(TRAIN, 6,'),
    ('N_HELDOUT = 16', 'N_HELDOUT = 4'), ('d.sample(n=min(300, len(d))', 'd.sample(n=min(20, len(d))'),
    ('"DEVICE": "cuda", "VAE_OFFLOAD": True, "DTYPE": "float16"', '"DEVICE": "cpu", "VAE_OFFLOAD": False, "DTYPE": "float32"'),
    ('"SCORER_MODELS": {}', f'"SCORER_MODELS": {{"clip_id": "{TM}/clip", "dino_id": "{TM}/dino"}}'),
    ('"--out", f"{OUT}/metrics"]', f'"--out", f"{{OUT}}/metrics", "--clip", "{TM}/clip", "--dino", "{TM}/dino", "--device", "cpu"]'),
    ('HO = g6.eval_heldout(OUT, HELDOUT_CONDS)',
     f'HO = g6.eval_heldout(OUT, HELDOUT_CONDS, device="cpu", models=gm.Models("cpu", "{TM}/clip", "{TM}/dino"))'),
    ('from huggingface_hub import hf_hub_download, HfFileSystem', 'from t6fake import hf_hub_download, HfFileSystem'),
    ('POLL = 300', 'POLL = 2'),
    ('MODEL_ID, WEIGHTS_DIR = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "/tmp/wan22_ti2v5b"',
     f'MODEL_ID, WEIGHTS_DIR = "x", "{SP}/day2/tiny_weights"'),
]


def run_cells():
    ns = {}
    for k, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = c["source"]
        if src.startswith("!pip"):
            continue
        if "UMT5EncoderModel" in src:
            E = {"__negative__": torch.randn(1, 16, 32).half(), "__empty__": torch.zeros(1, 16, 32).half()}
            for key in ns["PROMPTS"]:
                E[key] = torch.randn(1, 16, 32).half()
            torch.save(E, ns["CONFIG"]["EMBEDS"]); continue
        if "snapshot_download(MODEL_ID" in src:
            continue
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            open(first.split()[1], "w").write(body); continue
        for a, b in SUBS:
            src = src.replace(a, b)
        src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
        src = src.replace("N_GPU = torch.cuda.device_count()", "N_GPU = 2")
        print(f"--- cell {k}", flush=True)
        exec(compile(src, f"cell{k}", "exec"), ns)
    return ns


t0 = time.time()
ns = run_cells()
OUT = "/kaggle/working/genesis6"
print("queue:", ns["STATE"])
expect = ["first_fox30", "first_train60", "fox30_C_lora", "train60_C_lora", "fox30_select", "fox30_m3",
          "heldout_a", "heldout_b", "fox30_beat", "fox30_C"]
for j in expect:
    assert j in ns["STATE"][".done"], (j, ns["STATE"])
assert not ns["STATE"][".failed"], ns["STATE"]
log = json.load(open(f"{OUT}/logs/day6_runs_all.json"))
sel = [r for r in log if "chosen" in r]
assert sel and all(len(r["scores"]) == 2 for r in sel), sel[:1]
beat = [r for r in log if r.get("tag", "").startswith("fox30_C_anchor_lora_beat_c")]
want = [it.get("cfg", 3.5) for it in ns["ITEMS"]["fox30beat"][1:]]
assert [r["cfg"] for r in sorted(beat, key=lambda r: r["clip"])] == want, (beat, want)
from genesis_day6 import plan_script     # 30 s at full size: 2 clips per beat, boost only on each beat's first clip
sys.path.insert(0, HERE)
its, info = plan_script(**{k: ns["FOX"][k] for k in ("subject", "style", "beats", "opening", "template")}, seconds=30,
                        fps=24, num_frames=49, m=2, beat_cfg=5.0)
print("30 s plan:", info, [it.get("cfg", "-") for it in its])
assert info["clips"] == 17 and info["warning"] is None and sum("cfg" in it for it in its) == 8
assert all(r["lora"] for r in log if r.get("tag", "").startswith("fox30_C_anchor_lora_c"))
assert not any(r["lora"] for r in log if r.get("tag", "").startswith("fox30_C_anchor_c"))
m3 = json.load(open(f"{OUT}/manifests/fox30_C_anchor_lora_m3.json")); m2 = json.load(open(f"{OUT}/manifests/fox30_C_anchor_lora.json"))
print("m2 clips", m2["clip_lengths"], "| m3 clips", m3["clip_lengths"])
assert m3["clip_lengths"][1] == 13 - 9 and m2["clip_lengths"][1] == 13 - 5
import pandas as pd
T = pd.read_csv(f"{OUT}/metrics/day6_main.csv"); print(T[["video", "method", "start_end_id", "seam_speed_ratio_median"]])
assert len(T) == 6
H = pd.read_csv(f"{OUT}/metrics/heldout_per_clip.csv"); print(H.groupby("cond").size().to_dict())
assert len(H) == 4 * 4
assert set(json.load(open(f"{OUT}/heldout/clips.json"))[0]) >= {"id", "key", "caption", "file"}
for f in ["fox30_timeline.jpg", "train60_timeline.jpg", "fox30_speed.png", "heldout_bars.png", "heldout_example.jpg"]:
    assert os.path.exists(f"{OUT}/figures/{f}"), f
names = zipfile.ZipFile("/kaggle/working/genesis6_essentials.zip").namelist()
assert any(n.endswith("train60_C_anchor_lora_FULL.mp4") for n in names)
assert any(n.endswith("fox30_COMPARE_select.mp4") for n in names)
assert any(n.endswith("metrics/heldout_summary.md") for n in names) and any(n.endswith("select_log.csv") for n in names)
print(len(names), "files in essentials zip")
print(f"DAY 6 NOTEBOOK TEST PASSED ({time.time() - t0:.0f}s)")
