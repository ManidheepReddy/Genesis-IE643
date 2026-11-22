"""CPU test of genesis_app: tiny random Wan2.2-shaped model, tiny CLIP/DINO, real Day 5 outputs as the gallery."""
import os, shutil, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import genesis_lib as gl, genesis_metrics as gm, genesis_app as ga

G = gl.Genesis(f"{SP}/day2/tiny_weights", 64, 64, 13, 2, 5.0, 3.0, 8, device="cpu", vae_offload=False,
               dtype=torch.float32)
enc = lambda text: torch.randn(1, 16, 32)
M = gm.Models(device="cpu", clip_id=f"{SP}/day4/t_models/clip", dino_id=f"{SP}/day4/t_models/dino")
W = f"{HERE}/t_app"; shutil.rmtree(W, ignore_errors=True)
app = ga.App(G, enc, M, f"{SP}/day3/tiny_out3/lora/genesis_continuation_lora", W, gallery_roots=[f"{SP}/ess5"],
             presets={"tiny": (64, 64, 13, 2, 1)}, fps=8)
for label in ["C + LoRA (final method)", "Baseline: independent clips, hard cuts", "C + LoRA, m=3 (experimental: 9-frame context)"]:
    last = None
    for out in app.generate(ga.EXAMPLE, 3, label, "tiny", 7000):
        last = out
    video, status, table, plot, strip = last
    print(label, "|", status.splitlines()[-1])
    assert video and os.path.exists(video), status
    print(table.to_string(index=False))
    assert strip.ndim == 3 and plot is not None
    import json, glob
    man = json.load(open(glob.glob(f"{os.path.dirname(os.path.dirname(video))}/manifests/*.json")[0]))
    print("  clips", man["clip_lengths"], "cuts", man["cuts"])
    if label.startswith("Baseline"):
        assert len(man["cuts"]) == len(man["clip_lengths"]) - 1
    elif "m=3" in label:
        assert man["clip_lengths"][1] == 13 - 9
    else:
        assert len(man["cuts"]) == 1
# empty prompt and gallery
assert "at least one prompt" in list(app.generate("  \n", 3, "C + LoRA (final method)", "tiny", 1))[-1][1]
items = app.gallery_items(); print(len(items), "gallery items:", list(items)[:4])
v, t = app.gallery_show([k for k in items if "fox20b_C_anchor_lora_FULL" in k][0])
assert v and t is not None and len(t) > 5, t
demo = ga.build(app, open(f"{SP}/ess5/genesis5/metrics/summary.md").read(),
                sorted(glob.glob(f"{SP}/ess5/genesis5/metrics/figures/metrics_*.png")))
print(type(demo).__name__, "built")
print("APP TEST PASSED")
