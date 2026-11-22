"""End-to-end CPU test of the Day 4 notebook: runs its real cells (with a tiny random Wan2.2-shaped model,
tiny CLIP/DINO models and real Day 3 videos as the 'Day 3 input'), then checks a time-budget skip."""
import glob, json, os, shutil, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
D4 = f"{SP}/day4"
NB = f"{HERE}/t_nb.ipynb"
subprocess.run([sys.executable, f"{HERE}/make_day5.py", NB], check=True)
cells = json.load(open(NB))["cells"]

# fake Kaggle layout: /kaggle/input/<day3 notebook>/genesis3 (+ LoRA), empty /kaggle/working
shutil.rmtree("/kaggle", ignore_errors=True)
os.makedirs("/kaggle/working"); os.makedirs("/kaggle/input/genesis-ie643-day3")
shutil.copytree(f"{D4}/t_inputs/genesis3", "/kaggle/input/genesis-ie643-day3/genesis3")
shutil.copytree(f"{SP}/day3/tiny_out3/lora", "/kaggle/input/genesis-ie643-day3/genesis3/lora")

TM = f"{D4}/t_models"
SUBS = [
    ('HEIGHT, WIDTH, NUM_FRAMES, FPS = 480, 832, 49, 24', 'HEIGHT, WIDTH, NUM_FRAMES, FPS = 64, 64, 13, 8'),
    ('STEPS, GUIDANCE, FLOW_SHIFT, BASE_SEED = 30, 5.0, 3.0, 4000', 'STEPS, GUIDANCE, FLOW_SHIFT, BASE_SEED = 2, 5.0, 3.0, 4000'),
    ('assert len(ITEMS["fox20b"]) == n20, (len(ITEMS["fox20b"]), n20)', 'pass'),
    ('"DEVICE": "cuda", "VAE_OFFLOAD": True, "DTYPE": "float16"', '"DEVICE": "cpu", "VAE_OFFLOAD": False, "DTYPE": "float32"'),
    ('POLL = 300', 'POLL = 2'),
    ('METRIC_MODELS = []', f'METRIC_MODELS = ["--clip", "{TM}/clip", "--dino", "{TM}/dino", "--device", "cpu"]'),
    ('MODEL_ID, WEIGHTS_DIR = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "/tmp/wan22_ti2v5b"',
     f'MODEL_ID, WEIGHTS_DIR = "x", "{SP}/day2/tiny_weights"'),
]
def run_cells(stop_after_config=False):
    ns = {}
    for k, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = c["source"]
        if src.startswith("!pip"):
            continue
        if "UMT5EncoderModel" in src:
            E = {"__negative__": torch.randn(1, 16, 32).half(), "__empty__": torch.zeros(1, 16, 32).half()}
            for its in ns["ITEMS"].values():
                for it in its:
                    E[it["key"]] = torch.randn(1, 16, 32).half()
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
        if stop_after_config and "INPUT_RUNS" in ns:
            return ns
    return ns

shutil.rmtree("/kaggle/input/genesis-ie643-day3"); os.makedirs("/kaggle/input/other-dataset/x")
try:
    run_cells(); raise AssertionError("should have stopped")
except SystemExit as e:
    print("fail-fast OK:", str(e)[:80])
shutil.rmtree("/kaggle"); os.makedirs("/kaggle/working"); os.makedirs("/kaggle/input/genesis-ie643-day3")
shutil.copytree(f"{D4}/t_inputs/genesis3", "/kaggle/input/genesis-ie643-day3/genesis3")
shutil.copytree(f"{SP}/day3/tiny_out3/lora", "/kaggle/input/genesis-ie643-day3/genesis3/lora")
ns = {}
t0 = time.time()
ns = run_cells()

OUT = "/kaggle/working/genesis5"
q = os.listdir(f"{OUT}/queue")
done = sorted(f[:-5] for f in q if f.endswith(".done"))
print("done:", done)
for j in ["first_fox20b", "C_anchor_lora_fox20b", "C_anchor_fox20b", "C_anchor_lora_cutdemo", "lora"]:
    assert j in done, j
man = json.load(open(f"{OUT}/manifests/cutdemo_C_anchor_lora.json"))
print("cutdemo manifest:", man["clip_lengths"], man["boundaries"], man["cuts"])
assert len(man["cuts"]) == 2 and man["clip_lengths"][2] == 13 and man["clip_lengths"][1] == 13 - 5
log = [json.loads(l) for f in glob.glob(f"{OUT}/logs/runs_worker*.jsonl") for l in open(f)]
assert all(r["lora"] is False for r in log if r["tag"].startswith("fox20b_C_anchor_c"))
assert all(r["lora"] is True for r in log if r["tag"].startswith("fox20b_C_anchor_lora_c"))
assert not any("_text_c" in r["tag"] and r["lora"] for r in log)
import pandas as pd
V = pd.read_csv(f"{OUT}/metrics/per_video.csv")
print(V.groupby("set").size().to_dict())
assert set(V["set"]) == {"day3", "day5"} and len(V[V["set"] == "day5"]) == 3
assert os.path.exists("/kaggle/working/genesis5_metrics.zip") and os.path.exists("/kaggle/working/genesis5_essentials.zip")
assert os.path.exists(f"{OUT}/figures/fox20b_timeline.jpg") and os.path.exists(f"{OUT}/figures/cutdemo_seams.jpg")
assert not os.path.exists(f"{OUT}/frames") and not os.path.exists(f"{OUT}/latents")
import zipfile
names = zipfile.ZipFile("/kaggle/working/genesis5_essentials.zip").namelist()
assert any(n.endswith("fox20b_C_anchor_lora_FULL.mp4") for n in names) and not any("_c3.mp4" in n for n in names)
print(len(names), "files in essentials zip")

# time budget: a job that cannot finish before the deadline is skipped, not started
P = json.load(open(f"{OUT}/logs/day5_plan.json"))
os.makedirs(f"{OUT}/frames", exist_ok=True); os.makedirs(f"{OUT}/embeds", exist_ok=True)
torch.save({"__negative__": torch.randn(1, 16, 32), "late_0": torch.randn(1, 16, 32)}, P["config"]["EMBEDS"])
P["config"]["DEADLINE"] = time.time() + 300
P["jobs"] = [{"id": "late", "name": "late", "items": [{"key": "late_0", "seed": 1, "prompt": "x"}] * 3,
              "method": "C_anchor", "m": 2, "deps": []}]
json.dump(P, open(f"{OUT}/logs/plan_late.json", "w"))
subprocess.run([sys.executable, f"{OUT}/code/genesis_queue_worker.py", "--plan", f"{OUT}/logs/plan_late.json",
                "--worker", "9"], check=True, capture_output=True)
assert os.path.exists(f"{OUT}/queue/late.skipped") and open(f"{OUT}/queue/late.skipped").read() == "time budget"
print(f"DAY 4 NOTEBOOK TEST PASSED ({time.time() - t0:.0f}s)")
