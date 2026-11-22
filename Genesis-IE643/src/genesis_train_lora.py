"""genesis_train_lora.py — continuation LoRA for Wan2.2 TI2V-5B (Genesis, IE 643).

Goal: TI2V-5B was trained to condition on ONE clean first frame. Method B hands it m = 1..3 clean latent frames.
This LoRA teaches the model to continue real videos from a multi-frame context.

Data: real clips (OpenVid-1M, CC-BY-4.0) -> 33 frames at ~24 fps, 384x672 -> VAE latents (9 latent frames).
Objective (flow matching, per-token timesteps, loss only on the frames to be generated):
    m ~ {1,2,3};  sigma ~ shift(logit-normal);  eps ~ N(0, I)
    x_sigma = (1 - sigma) x0 + sigma eps   on frames m..T-1;   frames 0..m-1 stay clean (x0), timestep 0
    loss = || v_theta(x, t, text) - (eps - x0) ||^2   averaged over frames m..T-1 only
Base weights frozen (fp16); LoRA (rank r) on the attention projections of every block, trained in fp32.
"""
import argparse, json, math, os, random, sys, tempfile, time, traceback, zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
a = ap.parse_args()
plan = json.load(open(a.plan))
C = plan["config"]
T = C["TRAIN"]
OUT, Q, LORA_DIR = C["OUT"], f'{C["OUT"]}/queue', C["LORA_DIR"]
DEV = C.get("DEVICE", "cuda")
DTYPE = getattr(torch, C.get("DTYPE", "float16"))
os.makedirs(Q, exist_ok=True); os.makedirs(LORA_DIR, exist_ok=True)
LOG = f"{OUT}/logs/train_lora.jsonl"


def log(rec):
    print(rec, flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ data preparation (real clips -> latents)
def load_clip(path, n, H, W, target_fps=24):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(fps / target_fps)))
    frames, idx = [], 0
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = fr.shape[:2]
            s = max(H / h, W / w)
            fr = cv2.resize(fr, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA)
            y, x = (fr.shape[0] - H) // 2, (fr.shape[1] - W) // 2
            frames.append(cv2.cvtColor(fr[y:y + H, x:x + W], cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return np.stack(frames) if len(frames) == n else None


def prepare():
    from diffusers import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
    sel = json.load(open(T["SELECTION"]))
    vae = AutoencoderKLWan.from_pretrained(f'{C["WEIGHTS_DIR"]}/vae', torch_dtype=torch.float32).to(DEV)
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(DEV)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(DEV)
    keys, lats, t0 = [], [], time.time()
    zf = zipfile.ZipFile(T["ZIP"]) if T.get("ZIP") else None
    tmp = tempfile.mkdtemp()
    for i, r in enumerate(sel):
        try:
            if zf is not None:
                path = zf.extract(r["member"], tmp)
            else:
                path = r["path"]
            fr = load_clip(path, T["FRAMES"], T["H"], T["W"])
            if zf is not None:
                os.remove(path)
            if fr is None:
                continue
            vid = torch.from_numpy(fr).float().div(127.5).sub(1).permute(3, 0, 1, 2).unsqueeze(0).to(DEV)
            with torch.no_grad():
                z = retrieve_latents(vae.encode(vid), sample_mode="argmax")
            lats.append(((z - mean) * inv_std).half().cpu()); keys.append(r["key"])
        except Exception as e:
            print("skip", r.get("member", r.get("path")), repr(e)[:120], flush=True)
        if i % 25 == 0:
            print(f"prep {i}/{len(sel)} kept {len(lats)} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"keys": keys, "latents": lats}, T["PREPARED"])
    log({"phase": "prep", "kept": len(lats), "of": len(sel), "sec": round(time.time() - t0)})
    del vae; torch.cuda.empty_cache()


# ------------------------------------------------------------------ training
def train():
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file

    data = torch.load(T["PREPARED"])
    caps = torch.load(T["CAPTIONS"])       # key -> [n_tokens, 4096] fp16 (unpadded)
    keys, lats = data["keys"], data["latents"]
    assert len(lats) > 10, f"too few training clips ({len(lats)})"

    def text(key):
        e = caps[key].float()
        out = torch.zeros(512, e.shape[-1]); out[: e.shape[0]] = e[:512]
        return out.unsqueeze(0).to(DEV, DTYPE)

    try:
        tr = WanTransformer3DModel.from_pretrained(f'{C["WEIGHTS_DIR"]}/transformer', torch_dtype=DTYPE, device_map=DEV)
    except Exception:
        tr = WanTransformer3DModel.from_pretrained(f'{C["WEIGHTS_DIR"]}/transformer', torch_dtype=DTYPE).to(DEV)
    tr.requires_grad_(False)
    if DEV == "cuda":
        tr.enable_gradient_checkpointing()
    lcfg = {"r": T["RANK"], "lora_alpha": T["RANK"], "init_lora_weights": True,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]}
    tr.add_adapter(LoraConfig(**lcfg), adapter_name="genesis")
    params = []
    for n, p in tr.named_parameters():
        if "lora_" in n:
            p.data = p.data.float(); p.requires_grad_(True); params.append(p)
    n_lora = sum(p.numel() for p in params)
    json.dump(lcfg, open(f"{LORA_DIR}/lora_config.json", "w"), indent=2)

    start = 0
    state_p = f"{LORA_DIR}/state.json"
    if os.path.exists(state_p) and os.path.exists(f"{LORA_DIR}/lora.safetensors"):
        set_peft_model_state_dict(tr, load_file(f"{LORA_DIR}/lora.safetensors"), adapter_name="genesis")
        for p in params:
            p.data = p.data.float()
        start = json.load(open(state_p))["step"]
        print("resumed LoRA at step", start, flush=True)

    opt = torch.optim.AdamW(params, lr=T["LR"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda" and DTYPE == torch.float16))
    rng = random.Random(1234 + start)
    gen = torch.Generator().manual_seed(1234 + start)
    shift = T["SHIFT"]

    def save(step):
        sd = {k: v.detach().float().cpu().contiguous() for k, v in get_peft_model_state_dict(tr, adapter_name="genesis").items()}
        save_file(sd, f"{LORA_DIR}/lora.safetensors")
        json.dump({"step": step}, open(state_p, "w"))

    log({"phase": "train_start", "clips": len(lats), "lora_params_M": round(n_lora / 1e6, 2), "start": start})
    t0, losses = time.time(), []
    tr.train()
    done = start
    for step in range(start, T["STEPS"]):
        if (time.time() - t0) / 60 > T["MAX_MIN"]:
            log({"phase": "time_limit", "step": step}); break
        i = rng.randrange(len(lats))
        x0 = lats[i].to(DEV, torch.float32)                       # 1,48,T,h,w
        Tl, h, w = x0.shape[2:]
        m = rng.choices(T["M_CHOICES"], weights=T["M_PROBS"])[0]
        m = min(m, Tl - 1)
        u = torch.sigmoid(torch.randn(1, generator=gen)).item()  # logit-normal
        sigma = shift * u / (1 + (shift - 1) * u)
        eps = torch.randn(x0.shape, generator=gen).to(DEV)
        mask = torch.ones(1, 1, Tl, h, w, device=DEV); mask[:, :, :m] = 0
        xt = (1 - sigma) * x0 + sigma * eps
        inp = (1 - mask) * x0 + mask * xt
        ts = (mask[0, 0][:, ::2, ::2] * sigma * 1000.0).flatten().unsqueeze(0)
        emb = text("__empty__" if rng.random() < T["DROP_TEXT"] else keys[i])
        pred = tr(hidden_states=inp.to(DTYPE), timestep=ts,
                  encoder_hidden_states=emb, return_dict=False)[0].float()
        target = eps - x0
        loss = (((pred - target) ** 2) * mask).sum() / (mask.sum() * x0.shape[1])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
        done = step + 1
        if (step + 1) % 25 == 0:
            el = time.time() - t0
            log({"step": step + 1, "loss": round(float(np.mean(losses[-25:])), 4), "m": m,
                 "sec_per_step": round(el / (step + 1 - start), 2), "min": round(el / 60, 1)})
        if (step + 1) % T["CKPT_EVERY"] == 0:
            save(step + 1)
    save(done)
    log({"phase": "train_end", "steps_done": done, "min": round((time.time() - t0) / 60, 1),
         "final_loss_avg100": round(float(np.mean(losses[-100:])), 4) if losses else None})


try:
    if not os.path.exists(T["PREPARED"]):
        prepare()
    if os.path.exists(f"{LORA_DIR}/state.json") and json.load(open(f"{LORA_DIR}/state.json"))["step"] >= T["STEPS"]:
        print("LoRA already trained", flush=True)
    else:
        train()
    open(f"{Q}/lora.done", "w").write("ok")
except Exception:
    tb = traceback.format_exc()
    print(tb, flush=True)
    open(f"{Q}/lora.failed", "w").write(tb)
    sys.exit(1)
