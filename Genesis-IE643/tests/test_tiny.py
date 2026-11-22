"""Offline CPU test of genesis_lib + genesis_worker with a tiny random Wan2.2-shaped model."""
import json, os, shutil, subprocess, sys
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "tiny_weights")
OUT = os.path.join(HERE, "tiny_out")
shutil.rmtree(W, ignore_errors=True); shutil.rmtree(OUT, ignore_errors=True)
for d in ["embeds", "videos", "latents", "frames", "logs", "manifests"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

torch.manual_seed(0)
vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                       attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0, is_residual=True,
                       in_channels=12, out_channels=12, patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16,
                       latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
vae.save_pretrained(f"{W}/vae")
tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=48,
                           out_channels=48, text_dim=32, freq_dim=16, ffn_dim=64, num_layers=1, cross_attn_norm=True,
                           qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, rope_max_seq_len=1024)
tr.save_pretrained(f"{W}/transformer")
UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0,
                        num_train_timesteps=1000).save_pretrained(f"{W}/scheduler")
json.dump({"_class_name": "WanPipeline", "expand_timesteps": True}, open(f"{W}/model_index.json", "w"))

E = {"__negative__": torch.randn(1, 16, 32)}
for seq in ["s1", "s2"]:
    for i in range(3):
        E[f"{seq}_{i}"] = torch.randn(1, 16, 32)
torch.save(E, f"{OUT}/embeds/e.pt")

H, Wd, NF = 64, 64, 13   # 13 frames -> 4 latent frames, so m = 3 (K = 9) still leaves one generated frame
plan = {"config": {"OUT": OUT, "WEIGHTS_DIR": W, "EMBEDS": f"{OUT}/embeds/e.pt", "HEIGHT": H, "WIDTH": Wd,
                   "NUM_FRAMES": NF, "STEPS": 2, "GUIDANCE": 5.0, "FLOW_SHIFT": 3.0, "FPS": 8, "DEVICE": "cpu",
                   "VAE_OFFLOAD": False, "DTYPE": "float32"},
        "workers": {"0": [{"seq": "s1", "n_clips": 3, "seeds": [1, 2, 3],
                           "methods": [["baseline", 1], ["A_lastframe", 1], ["B_m2", 2], ["B_m3", 3]]}],
                    "1": [{"seq": "s2", "n_clips": 3, "seeds": [4, 5, 6], "methods": [["baseline", 1], ["B_m2", 2]]}]}}
json.dump(plan, open(f"{OUT}/plan.json", "w"))

for w in ["0", "1"]:
    r = subprocess.run([sys.executable, f"{HERE}/genesis_worker.py", "--plan", f"{OUT}/plan.json", "--worker", w],
                       capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-6000:])
    assert r.returncode == 0, f"worker {w} failed"

# checks
sys.path.insert(0, HERE)
import genesis_lib as gl
for name, m in [("baseline", 1), ("A_lastframe", 1), ("B_m2", 2), ("B_m3", 3)]:
    man = json.load(open(f"{OUT}/manifests/s1_{name}.json"))
    exp = [NF] + [NF if name == "baseline" else NF - gl.context_frames(m)] * 2
    assert man["clip_lengths"] == exp, (name, man["clip_lengths"], exp)
    full = gl.read_video(f"{OUT}/videos/s1_{name}_FULL.mp4")
    assert full.shape[0] == sum(exp), (name, full.shape, sum(exp))
    print(name, "OK", man)

# the conditioning latents must be exactly the re-encoded context (first m latent frames)
lat = torch.load(f"{OUT}/latents/s1_B_m2_c1.pt").float()
assert lat.shape == (1, 48, 4, H // 16, Wd // 16), lat.shape
print("latent shape OK", tuple(lat.shape))

# resume: rerun worker 0, everything should be resumed
r = subprocess.run([sys.executable, f"{HERE}/genesis_worker.py", "--plan", f"{OUT}/plan.json", "--worker", "0"],
                   capture_output=True, text=True)
assert r.returncode == 0 and "'resumed': True" in r.stdout and "'sec'" not in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
print("resume OK")
print("clips_for_duration(20s, 24fps, 49, m=2) =", gl.clips_for_duration(20, 24, 49, 2))
print("ALL TESTS PASSED")
