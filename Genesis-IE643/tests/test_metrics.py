"""Offline test of genesis_metrics.py on REAL Day 3 videos (cut out of the uploaded 2x3 COMPARE grids)
with tiny random CLIP/DINO models (HF downloads are blocked here). Pixel/colour/flow metrics are real;
CLIP/DINO numbers are meaningless in this test — they only check that the code path runs."""
import json, os, shutil, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import genesis_metrics as gm

U = "/root/.claude/uploads/fb2dfe20-4637-5c35-8a46-573f31a5ae53"
SRC = {"fox": f"{U}/316a9368-fox_COMPARE.mp4", "car": f"{U}/194aa8a5-car_COMPARE.mp4",
       "balloon": f"{U}/6dd16fa7-balloon_COMPARE.mp4", "boat": f"{U}/3e0d3d8b-boat_COMPARE.mp4"}
ROOT = f"{HERE}/t_inputs/genesis3"
OUT = f"{HERE}/t_metrics"
shutil.rmtree(f"{HERE}/t_inputs", ignore_errors=True); shutil.rmtree(OUT, ignore_errors=True)
for d in ("videos", "manifests", "logs"):
    os.makedirs(f"{ROOT}/{d}")

ORDER = ["baseline", "A_lastframe", "B_m2", "C_anchor", "B_m2_lora", "C_anchor_lora"]
M = {"baseline": 1, "A_lastframe": 1, "B_m2": 2, "C_anchor": 2, "B_m2_lora": 2, "C_anchor_lora": 2}
prompts = {s: [f"{s} prompt number {i}" for i in range(5)] for s in SRC}
json.dump({"config": {"FPS": 24}, "prompts": prompts}, open(f"{ROOT}/logs/day3_plan.json", "w"))
from diffusers.utils import export_to_video
for seq, path in SRC.items():
    g = gm.read_u8(path)
    H, W = g.shape[1] // 2, g.shape[2] // 3
    for k, meth in enumerate(ORDER):
        r, c = divmod(k, 3)
        drop = 0 if meth == "baseline" else (1 if M[meth] == 1 else 5)
        lengths = [49] + [49 - drop] * 4
        n = sum(lengths)
        tile = g[:n, r * H + 20:(r + 1) * H, c * W:(c + 1) * W]          # skip the label strip
        export_to_video(list(tile.astype(np.float32) / 255), f"{ROOT}/videos/{seq}_{meth}_FULL.mp4", fps=24)
        json.dump({"seq": seq, "method": meth, "m": M[meth], "clip_lengths": lengths,
                   "boundaries": [int(b) for b in np.cumsum(lengths)[:-1]], "lora": "lora" in meth,
                   "anchor": 0.7 if "C_" in meth else 0.0},
                  open(f"{ROOT}/manifests/{seq}_{meth}.json", "w"))
# one [CUT] video: fox baseline clips 0-1 + boat baseline clips 2-3, cut at 98, chained seam elsewhere
f = gm.read_u8(f"{ROOT}/videos/fox_C_anchor_lora_FULL.mp4")[:93]
b = gm.read_u8(f"{ROOT}/videos/boat_C_anchor_lora_FULL.mp4")[:93]
os.makedirs(f"{HERE}/t_inputs/genesis4/videos"); os.makedirs(f"{HERE}/t_inputs/genesis4/manifests")
os.makedirs(f"{HERE}/t_inputs/genesis4/logs")
export_to_video(list(np.concatenate([f, b]).astype(np.float32) / 255),
                f"{HERE}/t_inputs/genesis4/videos/cutdemo_C_anchor_lora_FULL.mp4", fps=24)
json.dump({"seq": "cutdemo", "method": "C_anchor_lora", "m": 2, "clip_lengths": [49, 44, 49, 44],
           "boundaries": [49, 93, 142], "cuts": [93], "prompts": ["a", "b", "c", "d"], "lora": True, "anchor": 0.7},
          open(f"{HERE}/t_inputs/genesis4/manifests/cutdemo_C_anchor_lora.json", "w"))
json.dump({"config": {"FPS": 24}}, open(f"{HERE}/t_inputs/genesis4/logs/day4_plan.json", "w"))


# ---- tiny random CLIP + DINO + a character-level CLIP tokenizer
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer, ViTConfig, ViTModel
TM = f"{HERE}/t_models"; shutil.rmtree(TM, ignore_errors=True); os.makedirs(f"{TM}/clip"); os.makedirs(f"{TM}/dino")
chars = list("abcdefghijklmnopqrstuvwxyz0123456789.,-'")
vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
for ch in chars:
    vocab.setdefault(ch, len(vocab)); vocab.setdefault(ch + "</w>", len(vocab))
json.dump(vocab, open(f"{TM}/clip/vocab.json", "w")); open(f"{TM}/clip/merges.txt", "w").write("#version: 0.2\n")
CLIPTokenizer(f"{TM}/clip/vocab.json", f"{TM}/clip/merges.txt").save_pretrained(f"{TM}/clip")
torch.manual_seed(0)
cfg = CLIPConfig(text_config=dict(vocab_size=len(vocab) + 2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, max_position_embeddings=77, bos_token_id=0, eos_token_id=1,
                                  pad_token_id=1),
                 vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                    image_size=224, patch_size=32), projection_dim=16)
CLIPModel(cfg).save_pretrained(f"{TM}/clip")
ViTModel(ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=224,
                   patch_size=16)).save_pretrained(f"{TM}/dino")

models = gm.Models(device="cpu", clip_id=f"{TM}/clip", dino_id=f"{TM}/dino")
V, C = gm.score_all([f"{HERE}/t_inputs"], OUT, models=models)
assert len(V) == 25, len(V)
cut = V[V["seq"] == "cutdemo"].iloc[0]
print("cutdemo:", cut[["n_shots", "seam_pix", "cut_pix"]].to_dict())
assert cut["n_shots"] == 2 and cut["cut_pix"] > cut["seam_pix"]
# second run must hit the cache (no model needed)
V2, _ = gm.score_all([f"{HERE}/t_inputs"], OUT, models=None)
assert len(V2) == 25
print(sorted(os.listdir(f"{OUT}/figures")))
print("METRICS TEST PASSED")
