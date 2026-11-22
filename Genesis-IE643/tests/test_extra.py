import json, os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import genesis_lib as gl
OUT = f"{HERE}/tiny_out"; plan = json.load(open(f"{OUT}/plan.json")); C = plan["config"]
G = gl.Genesis(C["WEIGHTS_DIR"], C["HEIGHT"], C["WIDTH"], C["NUM_FRAMES"], 2, 5.0, 3.0, 8, device="cpu", vae_offload=False, dtype=torch.float32)
E = torch.load(C["EMBEDS"])
# 1) conditioning held: first m latent frames of a chained clip == normalised encoding of the context
tail = gl.load_tail(OUT, "s1_baseline_c0")
for m in (1, 2, 3):
    K = gl.context_frames(m)
    _, lat = G.context_clip(tail[-K:], E, "s1_1", 7)
    vid = torch.from_numpy(tail[-K:]).float().div(127.5).sub(1).permute(3, 0, 1, 2).unsqueeze(0)
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
    ref = retrieve_latents(G.vae.encode(vid), sample_mode="argmax")
    diff = (lat[:, :, :m] - ref).abs().max().item()
    print(f"m={m}: context latents preserved, max diff {diff:.2e}"); assert diff < 1e-4
# 2) Method A through our path == stock TI2V single-image path
from PIL import Image
_, lat_ours = G.context_clip(tail[-1:], E, "s1_1", 7)
stock = gl.WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=G.vae, scheduler=G.i2v.scheduler, transformer=G.transformer, expand_timesteps=True)
lat_stock = stock(image=Image.fromarray(tail[-1]), prompt_embeds=E["s1_1"], **G._kwargs(E, 7)).frames
d = (lat_ours - lat_stock).abs().max().item(); print(f"Method A == stock TI2V: max diff {d:.2e}"); assert d < 1e-3
print("EXTRA TESTS PASSED")
