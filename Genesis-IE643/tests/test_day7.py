"""CPU end-to-end test of the novelty notebook: tiny random Wan2.2-shaped model, tiny CLIP/DINO, tiny random LoRA."""
import json, os, shutil, subprocess, sys, time, zipfile
import numpy as np
import torch

HERE = "/home/claude/g7"
FX = f"{HERE}/fixtures"
NB = f"{HERE}/t_nb.ipynb"
subprocess.run([sys.executable, f"{HERE}/make_day7.py", NB], check=True)
cells = json.load(open(NB))["cells"]

# ---- fixtures
if not os.path.exists(f"{FX}/done"):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer, ViTConfig, ViTModel
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    W = f"{FX}/tiny_weights"; os.makedirs(W, exist_ok=True)
    torch.manual_seed(0)
    AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                     temperal_downsample=[False, True, True], dropout=0.0, is_residual=True, in_channels=12, out_channels=12,
                     patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16, latents_mean=[0.0] * 48,
                     latents_std=[1.0] * 48).save_pretrained(f"{W}/vae")
    tr = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=48,
                               out_channels=48, text_dim=32, freq_dim=16, ffn_dim=64, num_layers=1, cross_attn_norm=True,
                               qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, rope_max_seq_len=1024)
    tr.save_pretrained(f"{W}/transformer")
    UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0,
                            num_train_timesteps=1000).save_pretrained(f"{W}/scheduler")
    json.dump({"_class_name": "WanPipeline", "expand_timesteps": True}, open(f"{W}/model_index.json", "w"))
    # LoRA
    L = f"{FX}/day3out/genesis3/lora/genesis_continuation_lora"; os.makedirs(L)
    lcfg = {"r": 4, "lora_alpha": 4, "init_lora_weights": "gaussian", "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]}
    tr.add_adapter(LoraConfig(**lcfg), adapter_name="genesis")
    save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(tr, adapter_name="genesis").items()}, f"{L}/lora.safetensors")
    json.dump(lcfg, open(f"{L}/lora_config.json", "w"))
    # CLIP / DINO
    TM = f"{FX}/t_models"; os.makedirs(f"{TM}/clip"); os.makedirs(f"{TM}/dino")
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789.,-'")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for ch in chars:
        vocab.setdefault(ch, len(vocab)); vocab.setdefault(ch + "</w>", len(vocab))
    json.dump(vocab, open(f"{TM}/clip/vocab.json", "w")); open(f"{TM}/clip/merges.txt", "w").write("#version: 0.2\n")
    CLIPTokenizer(f"{TM}/clip/vocab.json", f"{TM}/clip/merges.txt").save_pretrained(f"{TM}/clip")
    cfg = CLIPConfig(text_config=dict(vocab_size=len(vocab) + 2, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, max_position_embeddings=77, bos_token_id=0, eos_token_id=1,
                                      pad_token_id=1),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        image_size=224, patch_size=32), projection_dim=16)
    CLIPModel(cfg).save_pretrained(f"{TM}/clip")
    ViTModel(ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=224,
                       patch_size=16)).save_pretrained(f"{TM}/dino")
    open(f"{FX}/done", "w").write("1")
TM = f"{FX}/t_models"

shutil.rmtree("/kaggle", ignore_errors=True)
os.makedirs("/kaggle/working"); os.makedirs("/kaggle/input")
shutil.copytree(f"{FX}/day3out", "/kaggle/input/genesis-ie643-day3")

SUBS = [
    ('HEIGHT, WIDTH, NUM_FRAMES, FPS = 384, 672, 49, 24', 'HEIGHT, WIDTH, NUM_FRAMES, FPS = 64, 64, 13, 8'),
    ('STEPS, GUIDANCE, FLOW_SHIFT, SEED = 20, 5.0, 3.0, 7000', 'STEPS, GUIDANCE, FLOW_SHIFT, SEED = 2, 5.0, 3.0, 7000'),
    ('SECONDS = 20', 'SECONDS = 8'),
    ('"DEVICE": "cuda", "VAE_OFFLOAD": True, "DTYPE": "float16"', '"DEVICE": "cpu", "VAE_OFFLOAD": False, "DTYPE": "float32"'),
    ('"SCORER_MODELS": {}', f'"SCORER_MODELS": {{"clip_id": "{TM}/clip", "dino_id": "{TM}/dino"}}'),
    ('"--out", f"{OUT}/metrics"]', f'"--out", f"{{OUT}}/metrics", "--clip", "{TM}/clip", "--dino", "{TM}/dino", "--device", "cpu"]'),
    ('M = gm.Models(device="cuda" if torch.cuda.is_available() else "cpu")', f'M = gm.Models("cpu", "{TM}/clip", "{TM}/dino")'),
    ('GEN_HOURS, SEC_PER_CLIP, POLL = 3.0, 180, 120', 'GEN_HOURS, SEC_PER_CLIP, POLL = 3.0, 180, 2'),
    ('(0, 4, 8, 12, 16, 20)', '(0, 2, 4, 6, 8, 9)'),
    ('MODEL_ID, WEIGHTS_DIR = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "/tmp/wan22_ti2v5b"',
     f'MODEL_ID, WEIGHTS_DIR = "x", "{FX}/tiny_weights"'),
]
ns = {}
t0 = time.time()
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
        open(first.split()[1].replace("{CODE}", ns["CODE"]), "w").write(body); continue
    for a, b in SUBS:
        assert a in src or not any(x in src for x in [a[:12]]), a
        src = src.replace(a, b)
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
    src = src.replace("N_GPU = torch.cuda.device_count()", f"N_GPU = {os.environ.get('T_NGPU', '2')}")
    print(f"--- cell {k}", flush=True)
    exec(compile(src, f"cell{k}", "exec"), ns)

OUT = "/kaggle/working/genesis7"
st = ns["STATE"]; print(st)
need = {"first", "C", "C_lora", "fix_m3", "fix_select"} if os.environ.get("T_NGPU", "2") == "2" else {"first", "C", "C_lora"}
assert set(st[".done"]) >= need and not st[".failed"], st
if os.environ.get("T_NGPU") == "1":
    assert len([x for x in st[".done"] if x.startswith("fix_")]) == 1, st
m2 = json.load(open(f"{OUT}/manifests/fox20_C_anchor.json")); m2l = json.load(open(f"{OUT}/manifests/fox20_C_anchor_lora.json"))
assert m2["clip_lengths"] == m2l["clip_lengths"] and not m2["lora"] and m2l["lora"], (m2["clip_lengths"], m2l["clip_lengths"])
v1, v2 = ns["VID"]["C_anchor"], ns["VID"]["C_anchor_lora"]
assert np.abs(v1[:13].astype(int) - v2[:13].astype(int)).max() <= 2, "first clip must be shared"   # same seed + prompt
assert np.abs(v1[13:].astype(int) - v2[13:].astype(int)).mean() > 0.5, "LoRA must change the continuation"
print("findings:", json.load(open(f"{OUT}/metrics/findings.json"))["_decision"])
T = ns["T"]; print(T.round(3).to_string()); assert len(T) == (4 if os.environ.get("T_NGPU", "2") == "2" else 3)
rows = ns["ROWS"]["C_anchor_lora"]; assert len(rows) == len(m2l["boundaries"])
names = zipfile.ZipFile("/kaggle/working/genesis7_novelty.zip").namelist()
for need in ["fox20_ALL_RUNS_JOINS_MARKED.mp4", "fox20_C_anchor_JOINS.mp4", "joins_curves.png", "join_frames.jpg",
             "timeline.jpg", "speed_timeline.png", "novelty_table.md", "findings.json",
             "joins_C_anchor_lora.csv"] + (["select_log.csv"] if os.environ.get("T_NGPU", "2") == "2" else []):
    assert any(n.endswith(need) for n in names), need
# the join marker really is on the first frame after a join
mk = ns["MK"]["C_anchor"]; b = m2["boundaries"][0]
assert mk[b, 2, 2, 0] > 200 and mk[b, 2, 2, 1] < 60, mk[b, 2, 2]
print(f"NOVELTY NOTEBOOK TEST PASSED ({time.time() - t0:.0f}s)")
