"""Offline CPU test for Day 3: real-clip prep from a zip, LoRA training, queue workers, Method C, LoRA switching."""
import json, os, shutil, subprocess, sys, zipfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(os.path.dirname(HERE), "day2", "tiny_weights")      # tiny random Wan2.2-shaped model from Day 2 test
OUT = os.path.join(HERE, "tiny_out3")
shutil.rmtree(OUT, ignore_errors=True)
for d in ["embeds", "videos", "latents", "frames", "logs", "manifests", "queue", "train", "lora"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
sys.path.insert(0, HERE)
import genesis_lib as gl
from diffusers.utils import export_to_video

# --- synthetic "real" training videos in a zip (moving square, 30 fps, 48 frames)
src = f"{OUT}/train/src"; os.makedirs(src, exist_ok=True)
rng = np.random.default_rng(0)
sel, caps = [], {"__empty__": torch.zeros(1, 32).half()}
for k in range(14):
    col = rng.uniform(0.2, 1, 3)
    fr = np.zeros((48, 72, 120, 3), np.float32)
    for t in range(48):
        x = 5 + t
        fr[t, 20:40, x:x + 20] = col
    export_to_video(list(fr), f"{src}/clip{k}.mp4", fps=30)
    sel.append({"member": f"vids/clip{k}.mp4", "key": f"cap_{k}"})
    caps[f"cap_{k}"] = torch.randn(7, 32).half()
with zipfile.ZipFile(f"{OUT}/train/part.zip", "w") as z:
    for k in range(14):
        z.write(f"{src}/clip{k}.mp4", f"vids/clip{k}.mp4")
sel.append({"member": "vids/missing.mp4", "key": "cap_0"})          # must be skipped gracefully
json.dump(sel, open(f"{OUT}/train/selection.json", "w"))
torch.save(caps, f"{OUT}/train/caption_embeds.pt")

E = {"__negative__": torch.randn(1, 16, 32)}
for i in range(3):
    E[f"s1_{i}"] = torch.randn(1, 16, 32)
torch.save(E, f"{OUT}/embeds/e.pt")

H, Wd, NF = 64, 64, 13
LORA = f"{OUT}/lora/genesis_continuation_lora"
cfg = {"OUT": OUT, "WEIGHTS_DIR": W, "EMBEDS": f"{OUT}/embeds/e.pt", "HEIGHT": H, "WIDTH": Wd, "NUM_FRAMES": NF,
       "STEPS": 2, "GUIDANCE": 5.0, "FLOW_SHIFT": 3.0, "FPS": 8, "DEVICE": "cpu", "VAE_OFFLOAD": False,
       "DTYPE": "float32", "LORA_DIR": LORA,
       "TRAIN": {"SELECTION": f"{OUT}/train/selection.json", "ZIP": f"{OUT}/train/part.zip",
                 "CAPTIONS": f"{OUT}/train/caption_embeds.pt", "PREPARED": f"{OUT}/train/latents.pt",
                 "H": 64, "W": 96, "FRAMES": 13, "STEPS": 30, "RANK": 4, "LR": 1e-3, "MAX_MIN": 10,
                 "SHIFT": 3.0, "M_CHOICES": [1, 2, 3], "M_PROBS": [0.3, 0.4, 0.3], "DROP_TEXT": 0.1,
                 "CKPT_EVERY": 10}}
base = {"seq": "s1", "n_clips": 3, "seeds": [1, 2, 3]}
jobs = [dict(base, id="base_s1", method="baseline", m=1, deps=[]),
        dict(base, id="B_s1", method="B_m2", m=2, deps=["base_s1"]),
        dict(base, id="C_s1", method="C_anchor", m=2, cfg_cont=3.5, anchor=0.7, deps=["base_s1"]),
        dict(base, id="Blora_s1", method="B_m2_lora", m=2, lora=True, deps=["base_s1", "lora"]),
        dict(base, id="Clora_s1", method="C_anchor_lora", m=2, cfg_cont=3.5, anchor=0.7, lora=True,
             deps=["base_s1", "lora"])]
json.dump({"config": cfg, "jobs": jobs}, open(f"{OUT}/plan.json", "w"))


def run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "torchvision" not in l and "warn" not in l.lower())
    print(out[-2500:])
    return r.returncode


# worker 1 first: runs base/B/C, then must exit when only LoRA jobs remain and lora has FAILED -> simulate order:
assert run([f"{HERE}/genesis_train_lora.py", "--plan", f"{OUT}/plan.json"]) == 0, "training failed"
assert os.path.exists(f"{OUT}/queue/lora.done")
prep = torch.load(f"{OUT}/train/latents.pt")
assert len(prep["latents"]) == 14 and tuple(prep["latents"][0].shape) == (1, 48, 4, 4, 6), (len(prep["latents"]), prep["latents"][0].shape)
tl = [json.loads(l) for l in open(f"{OUT}/logs/train_lora.jsonl")]
print("train log:", tl[-1])
assert run([f"{HERE}/genesis_queue_worker.py", "--plan", f"{OUT}/plan.json", "--worker", "1"]) == 0
assert run([f"{HERE}/genesis_queue_worker.py", "--plan", f"{OUT}/plan.json", "--worker", "0"]) == 0
for j in jobs:
    assert os.path.exists(f"{OUT}/queue/{j['id']}.done"), j["id"]
for meth in ["baseline", "B_m2", "C_anchor", "B_m2_lora", "C_anchor_lora"]:
    man = json.load(open(f"{OUT}/manifests/s1_{meth}.json")); print(meth, man["clip_lengths"], man["lora"], man["anchor"])

# LoRA must change the output, and switching it off must restore the base model exactly
la, lb = torch.load(f"{OUT}/latents/s1_B_m2_c1.pt").float(), torch.load(f"{OUT}/latents/s1_B_m2_lora_c1.pt").float()
d_lora = (la - lb).abs().max().item(); print("B vs B+LoRA latent diff", d_lora); assert d_lora > 1e-4
G = gl.Genesis(W, H, Wd, NF, 2, 5.0, 3.0, 8, device="cpu", vae_offload=False, dtype=torch.float32)
ctx = gl.load_tail(OUT, "s1_baseline_c0")[-5:]
_, l0 = G.context_clip(ctx, E, "s1_1", 2)
G.set_lora(LORA); _, l1 = G.context_clip(ctx, E, "s1_1", 2)
G.set_lora(None); _, l2 = G.context_clip(ctx, E, "s1_1", 2)
print("off/on diff", (l0 - l1).abs().max().item(), "off/off-again diff", (l0 - l2).abs().max().item())
assert (l0 - l2).abs().max().item() == 0 and (l0 - l1).abs().max().item() > 1e-4

# colour anchoring pulls stats toward the reference
f = np.clip(rng.uniform(0.6, 1.0, (3, 32, 32, 3)).astype(np.float32), 0, 1)
ref = gl.lab_stats(gl.to_uint8(rng.uniform(0.0, 0.5, (3, 32, 32, 3)).astype(np.float32)))
before = np.abs(gl.lab_stats(gl.to_uint8(f))[0] - ref[0]).mean()
after = np.abs(gl.lab_stats(gl.to_uint8(gl.colour_anchor(f, ref, 0.7)))[0] - ref[0]).mean()
print(f"anchor: LAB mean distance {before:.1f} -> {after:.1f}"); assert after < 0.5 * before

# failure path: if lora failed, LoRA jobs are skipped and workers exit
shutil.rmtree(f"{OUT}/queue"); os.makedirs(f"{OUT}/queue")
for j in ["base_s1", "B_s1", "C_s1"]:
    open(f"{OUT}/queue/{j}.done", "w").write("x")
open(f"{OUT}/queue/lora.failed", "w").write("x")
assert run([f"{HERE}/genesis_queue_worker.py", "--plan", f"{OUT}/plan.json", "--worker", "0"]) == 0
assert os.path.exists(f"{OUT}/queue/Blora_s1.skipped") and os.path.exists(f"{OUT}/queue/Clora_s1.skipped")
print("DAY 3 TESTS PASSED")
