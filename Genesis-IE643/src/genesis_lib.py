"""
genesis_lib.py — Genesis (IE 643): long-form video by chaining short Wan2.2 TI2V-5B clips.

Chaining methods
  baseline : every clip generated independently from text (T2V) and joined with hard cuts.
  A (m=1)  : clip k+1 is conditioned on the LAST FRAME of clip k (the model's own TI2V first-frame conditioning).
  B (m>=2) : clip k+1 is conditioned on the last K = 1 + 4(m-1) FRAMES of clip k. They are re-encoded by the
             causal 3D VAE into m clean latent frames, placed at the start of the new clip's latent; those tokens
             get flow-matching timestep 0 (clean) and are re-injected after every solver step, so the model
             denoises the rest of the clip around a context that shows the motion, not just a still frame.
  C        : Method B + drift control. (i) every generated frame is colour-anchored to clip 0 (its LAB mean and
             standard deviation are pulled toward clip 0's by a factor `anchor`), so exposure and saturation
             cannot creep upward clip after clip; (ii) continuation clips use a lower CFG scale (`cfg_cont`),
             because guidance applied on already-saturated context compounds the drift.
  +LoRA    : any chained method with the continuation LoRA (genesis_train_lora.py) switched on.

Flow matching (what the model was trained with)
  x_s = (1 - s) * x_0 + s * eps,  s in [0, 1]   (s = 1 pure noise, s = 0 clean)
  model predicts velocity v = eps - x_0;  loss = || (eps - x_0) - v_theta(x_s, s, text) ||^2
  TI2V-5B gives every latent token its own timestep: conditioning tokens get s = 0, the rest the current s.
"""
import json, os, time

import numpy as np
import torch
import diffusers
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import get_module_execution_device, randn_tensor
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from PIL import Image

TEMPORAL = 4      # Wan2.2 VAE temporal compression
TAIL_KEEP = 9     # pixel frames kept losslessly per clip = context for m <= 3
SAVE_LATENTS = True   # Day 4+: set False to keep the notebook output small


def context_frames(m: int) -> int:
    """Pixel frames the VAE maps to m latent frames (first frame alone, then groups of 4)."""
    return 1 + TEMPORAL * (m - 1)


def new_frames_per_clip(num_frames: int, m: int) -> int:
    """Frames a chained clip adds to the video (its first K frames repeat the context and are dropped)."""
    return num_frames - context_frames(m)


def clips_for_duration(seconds: float, fps: int, num_frames: int, m: int) -> int:
    """Number of clips needed so the stitched video lasts at least `seconds`."""
    need = max(int(round(seconds * fps)), num_frames)
    return 1 + int(np.ceil((need - num_frames) / new_frames_per_clip(num_frames, m)))


def _prepare_latents_ctx(self, image, batch_size, num_channels_latents=48, height=480, width=832,
                         num_frames=49, dtype=None, device=None, generator=None, latents=None, last_image=None):
    """Replacement for WanImageToVideoPipeline.prepare_latents (expand_timesteps / TI2V path).

    If self._genesis_ctx (uint8 array, K x H x W x 3) is set, it is encoded into m = (K-1)/4 + 1 clean latent
    frames and the first m latent frames are masked as 'known' (timestep 0, re-injected every step).
    Otherwise falls back to the stock single-image behaviour.
    """
    ctx = getattr(self, "_genesis_ctx", None)
    if ctx is None:
        return WanImageToVideoPipeline.prepare_latents(self, image, batch_size, num_channels_latents, height, width,
                                                       num_frames, dtype, device, generator, latents, last_image)
    K = ctx.shape[0]
    assert (K - 1) % TEMPORAL == 0, f"context length {K} must be 1 + 4(m-1)"
    assert ctx.shape[1:3] == (height, width), f"context frames {ctx.shape[1:3]} != ({height}, {width})"

    T = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    h, w = height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial
    shape = (batch_size, num_channels_latents, T, h, w)
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)

    vid = torch.from_numpy(np.ascontiguousarray(ctx)).float().div(127.5).sub(1.0)   # K,H,W,3 in [-1, 1]
    vid = vid.permute(3, 0, 1, 2).unsqueeze(0)                                     # 1,3,K,H,W
    vae_dev = get_module_execution_device(self.vae)
    with torch.no_grad():
        lc = retrieve_latents(self.vae.encode(vid.to(vae_dev, self.vae.dtype)), sample_mode="argmax")
    lc = lc.to(device=device, dtype=dtype)
    mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
    inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
    lc = (lc - mean) * inv_std

    m = lc.shape[2]
    assert m < T, "context must be shorter than the clip"
    cond = torch.zeros_like(latents)
    cond[:, :, :m] = lc.expand(batch_size, -1, -1, -1, -1)
    mask = torch.ones(1, 1, T, h, w, dtype=dtype, device=device)
    mask[:, :, :m] = 0
    if getattr(self, "_genesis_vae_offload", False):
        self.vae.to("cpu")
        torch.cuda.empty_cache()
    return latents, cond, mask


class _T2V(WanPipeline):
    """Stock T2V pipeline, but always computes on the GPU even while the VAE sits on the CPU."""
    @property
    def _execution_device(self):
        return torch.device(self._genesis_device)


class _CtxI2V(WanImageToVideoPipeline):
    """TI2V pipeline whose conditioning can be a multi-frame context (see _prepare_latents_ctx)."""
    _genesis_ctx = None
    _genesis_vae_offload = False
    prepare_latents = _prepare_latents_ctx

    @property
    def _execution_device(self):
        return torch.device(self._genesis_device)


class Genesis:
    """Holds one TI2V-5B transformer + VAE on one GPU and generates / chains clips."""

    def __init__(self, weights_dir, height=480, width=832, num_frames=49, steps=30, guidance=5.0,
                 flow_shift=3.0, fps=24, device="cuda", vae_offload=True, dtype=torch.float16):
        self.h, self.w, self.nf, self.steps, self.cfg, self.fps = height, width, num_frames, steps, guidance, fps
        self.device, self.vae_offload, self.dtype = device, vae_offload, dtype
        idx = json.load(open(f"{weights_dir}/model_index.json"))
        expand = bool(idx.get("expand_timesteps", False))
        assert expand, "TI2V-5B must load with expand_timesteps=True (per-token timesteps)"

        self.vae = AutoencoderKLWan.from_pretrained(f"{weights_dir}/vae", torch_dtype=torch.float32)
        self.vae.enable_tiling()
        self.vae.to("cpu" if vae_offload else device)
        try:
            self.transformer = WanTransformer3DModel.from_pretrained(
                f"{weights_dir}/transformer", torch_dtype=dtype, device_map=device)
        except Exception as e:  # older/newer API without device_map support for this model
            print("device_map load failed, loading on CPU first:", repr(e)[:200])
            self.transformer = WanTransformer3DModel.from_pretrained(
                f"{weights_dir}/transformer", torch_dtype=dtype).to(device)
        self.transformer.eval()

        cfg = json.load(open(f"{weights_dir}/scheduler/scheduler_config.json"))
        cls = getattr(diffusers, cfg["_class_name"])
        sched = cls.from_pretrained(f"{weights_dir}/scheduler")
        key = "flow_shift" if "flow_shift" in sched.config else ("shift" if "shift" in sched.config else None)
        if key:
            sched = cls.from_config(sched.config, **{key: flow_shift})

        common = dict(tokenizer=None, text_encoder=None, vae=self.vae, scheduler=sched,
                      transformer=self.transformer, expand_timesteps=expand)
        self.t2v = _T2V(**common)
        self.i2v = _CtxI2V(**common)
        self.i2v._genesis_vae_offload = vae_offload
        for p in (self.t2v, self.i2v):
            p._genesis_device = device
            p.set_progress_bar_config(disable=True)
        self._lora_path = None      # adapter loaded into the transformer (None = never loaded)
        self.lora_active = False

    # ---------------------------------------------------------------- LoRA
    def set_lora(self, path):
        """path=None switches the adapter off; a directory with lora.safetensors + lora_config.json switches it on."""
        if path is None:
            if self._lora_path is not None and self.lora_active:
                self.transformer.disable_adapters()
            self.lora_active = False
            return
        if self._lora_path is None:
            from peft import LoraConfig, set_peft_model_state_dict
            from safetensors.torch import load_file
            cfg = json.load(open(f"{path}/lora_config.json"))
            self.transformer.add_adapter(LoraConfig(**cfg), adapter_name="genesis")
            sd = load_file(f"{path}/lora.safetensors")
            res = set_peft_model_state_dict(self.transformer, sd, adapter_name="genesis")
            missing = [k for k in getattr(res, "unexpected_keys", [])]
            assert not missing, f"LoRA keys not loaded: {missing[:5]}"
            for n, prm in self.transformer.named_parameters():   # cast only the adapter: some base modules
                if "lora_" in n:                                    # (norms, time embedder) must stay fp32
                    prm.data = prm.data.to(self.dtype)
            self._lora_path = path
        elif path != self._lora_path:
            raise ValueError("a different LoRA is already loaded in this process")
        self.transformer.enable_adapters()
        self.lora_active = True

    # ---------------------------------------------------------------- helpers
    def _emb(self, E, key):
        return E[key].to(self.device, self.dtype)

    def _kwargs(self, E, seed, guidance=None):
        return dict(negative_prompt_embeds=self._emb(E, "__negative__"), height=self.h, width=self.w,
                    num_frames=self.nf, num_inference_steps=self.steps,
                    guidance_scale=self.cfg if guidance is None else guidance,
                    generator=torch.Generator("cpu").manual_seed(int(seed)), output_type="latent")

    @torch.no_grad()
    def _decode(self, lat):
        v = self.vae
        v.to(self.device)
        try:
            z = lat.to(self.device, v.dtype)
            mean = torch.tensor(v.config.latents_mean).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            std = torch.tensor(v.config.latents_std).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            video = v.decode(z * std + mean, return_dict=False)[0]
            return self.t2v.video_processor.postprocess_video(video, output_type="np")[0]  # F,H,W,3 in [0,1]
        finally:
            if self.vae_offload:
                v.to("cpu")
            torch.cuda.empty_cache()

    def decode(self, lat):
        try:
            return self._decode(lat)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            try:
                return self._decode(lat)
            finally:
                self.transformer.to(self.device)

    # ---------------------------------------------------------------- generation
    def text_clip(self, E, key, seed):
        """Independent clip from text only (baseline, and clip 0 of every chain)."""
        lat = self.t2v(prompt_embeds=self._emb(E, key), **self._kwargs(E, seed)).frames
        return self.decode(lat), lat

    def context_clip(self, ctx_uint8, E, key, seed, guidance=None):
        """Clip conditioned on K = 1 + 4(m-1) previous frames (m = 1 is Method A, m >= 2 is Method B)."""
        if self.vae_offload:
            self.vae.to(self.device)
        self.i2v._genesis_ctx = ctx_uint8
        try:
            dummy = Image.fromarray(ctx_uint8[-1])   # required argument; real conditioning is _genesis_ctx
            lat = self.i2v(image=dummy, prompt_embeds=self._emb(E, key), **self._kwargs(E, seed, guidance)).frames
        finally:
            self.i2v._genesis_ctx = None
            if self.vae_offload:
                self.vae.to("cpu")
        return self.decode(lat), lat


# -------------------------------------------------------------------- storage
def to_uint8(frames):
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def frames_ok(frames):
    return bool(np.isfinite(frames).all() and frames.max() > 0.02)


def save_clip(out, tag, frames, lat, fps):
    export_to_video(list(frames), f"{out}/videos/{tag}.mp4", fps=fps)
    if SAVE_LATENTS:
        torch.save(lat.detach().cpu().half(), f"{out}/latents/{tag}.pt")
    np.save(f"{out}/frames/{tag}_tail.npy", to_uint8(frames[-TAIL_KEEP:]))


def clip_done(out, tag):
    return os.path.exists(f"{out}/frames/{tag}_tail.npy") and os.path.exists(f"{out}/videos/{tag}.mp4")


def load_tail(out, tag):
    return np.load(f"{out}/frames/{tag}_tail.npy")


def lab_stats(frames_u8):
    """Mean and std of each LAB channel over a stack of uint8 RGB frames (OpenCV LAB, 0-255 scale)."""
    import cv2
    lab = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in frames_u8]).reshape(-1, 3).astype(np.float32)
    return lab.mean(0), lab.std(0) + 1e-6


def colour_anchor(frames, ref, strength):
    """Pull each frame's LAB mean/std toward the reference stats by `strength` in [0, 1]. frames: float [0,1]."""
    import cv2
    ref_mu, ref_sd = ref
    out = np.empty_like(frames)
    for j, f in enumerate(to_uint8(frames)):
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32)
        mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        t_mu, t_sd = mu + strength * (ref_mu - mu), sd + strength * (ref_sd - sd)
        lab = (lab - mu) / sd * t_sd + t_mu
        rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        out[j] = rgb.astype(np.float32) / 255.0
    return out


def read_video(path):
    import imageio.v3 as iio
    return np.stack(list(iio.imiter(path))).astype(np.float32) / 255.0


def log_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# -------------------------------------------------------------------- one sequence, one method
def run_sequence(G, E, out, seq, n_clips, seeds, method, m, log_path, cfg_cont=None, anchor=0.0, lora=None):
    """method: 'baseline' or a chain name; m: context latent frames (1 = Method A);
    cfg_cont: CFG scale for continuation clips (None = same as clip 0); anchor: colour-anchoring strength;
    lora: LoRA directory to switch on for the continuation clips (None = off)."""
    clips = []
    ref = None
    for i in range(n_clips):
        if method == "baseline" or i == 0:
            tag = f"{seq}_baseline_c{i}"
        else:
            tag = f"{seq}_{method}_c{i}"
        if clip_done(out, tag):
            frames = read_video(f"{out}/videos/{tag}.mp4")
            stats = {"resumed": True}
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if method == "baseline" or i == 0:
                G.set_lora(None)                     # clip 0 is always the shared, unmodified baseline clip
                frames, lat = G.text_clip(E, f"{seq}_{i}", seeds[i])
            else:
                G.set_lora(lora)
                K = context_frames(m)
                ctx = load_tail(out, prev_tag)[-K:]
                frames, lat = G.context_clip(ctx, E, f"{seq}_{i}", seeds[i], guidance=cfg_cont)
                if anchor > 0:
                    if ref is None:
                        ref = lab_stats(to_uint8(read_video(f"{out}/videos/{seq}_baseline_c0.mp4")))
                    frames = np.concatenate([frames[:K], colour_anchor(frames[K:], ref, anchor)], axis=0)
            stats = {"sec": round(time.time() - t0, 1),
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
                     "frames_ok": frames_ok(frames)}
            save_clip(out, tag, frames, lat, G.fps)
        drop = 0 if (method == "baseline" or i == 0) else context_frames(m)
        clips.append(frames[drop:])
        prev_tag = tag
        rec = {"seq": seq, "method": method, "m": m, "clip": i, "tag": tag, "cfg_cont": cfg_cont,
               "anchor": anchor, "lora": bool(lora) and i > 0, **stats}
        log_jsonl(log_path, rec)
        print(rec, flush=True)

    full = np.concatenate(clips, axis=0)
    export_to_video(list(full), f"{out}/videos/{seq}_{method}_FULL.mp4", fps=G.fps)
    lengths = [int(c.shape[0]) for c in clips]
    G.set_lora(None)
    json.dump({"seq": seq, "method": method, "m": m, "cfg_cont": cfg_cont, "anchor": anchor, "lora": bool(lora),
               "clip_lengths": lengths,
               "boundaries": [int(b) for b in np.cumsum(lengths)[:-1]], "seconds": round(full.shape[0] / G.fps, 2)},
              open(f"{out}/manifests/{seq}_{method}.json", "w"), indent=2)
    return full


# -------------------------------------------------------------------- a "script": several shots separated by [CUT]
def parse_script(prompts):
    """['p0', 'p1', '[CUT]', 'p2', ...] -> [{'prompt': 'p0', 'cut': False}, {'prompt': 'p1', 'cut': False},
    {'prompt': 'p2', 'cut': True}, ...]. A prompt may also start with '[CUT]' itself."""
    items, pending_cut = [], False
    for p in prompts:
        p = p.strip()
        while p.upper().startswith("[CUT]"):
            pending_cut, p = True, p[5:].strip()
        if not p:
            continue
        items.append({"prompt": p, "cut": bool(items) and pending_cut})
        pending_cut = False
    return items


def expand_script(items, n_clips):
    """Spread the script's prompts over n_clips clips: each shot keeps its prompts in order; when there are more
    clips than prompts, the clips are shared out as evenly as possible (each prompt repeats for a few clips)."""
    n = len(items)
    if n_clips <= n:
        return [dict(it) for it in items[:max(n_clips, 1)]]
    reps = [n_clips // n + (1 if i < n_clips % n else 0) for i in range(n)]
    out = []
    for it, r in zip(items, reps):
        for k in range(r):
            out.append(dict(it, cut=it["cut"] and k == 0))
    return out


def run_script(G, E, out, name, items, method, m, log_path, cfg_cont=None, anchor=0.0, lora=None, finalize=True,
               n_cand=1, scorer=None):
    """Generalised run_sequence. items: [{'key': embedding key, 'seed': int, 'cut': bool, 'prompt': str}, ...].
    A clip with cut=True (and clip 0) starts a NEW SHOT: generated from text only (base model), giving a hard cut.
    Every other clip continues the previous one with the chosen method (m context latent frames, CFG for
    continuation clips, colour anchoring to the first clip of the CURRENT shot, optional LoRA).
    Shot-start clips are tagged {name}_text_c{i} so that different methods share them.
    Day 6: an item may carry its own 'cfg' (overrides cfg_cont for that clip). With n_cand > 1 every continuation
    clip is generated n_cand times (seeds seed, seed + 100000, ...) and `scorer` keeps the best candidate
    (generate-and-select); the rejected candidates are saved as {tag}_rej{k}.mp4 for inspection."""
    clips, cuts, ref, prev_tag, pos = [], [], None, None, 0
    for i, it in enumerate(items):
        new_shot = i == 0 or bool(it.get("cut"))
        tag = f"{name}_text_c{i}" if new_shot else f"{name}_{method}_c{i}"
        if new_shot:
            ref = None
            shot_first = tag
            if i > 0:
                cuts.append(pos)
        if clip_done(out, tag):
            frames = read_video(f"{out}/videos/{tag}.mp4")
            stats = {"resumed": True}
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if new_shot:
                G.set_lora(None)
                frames, lat = G.text_clip(E, it["key"], it["seed"])
            else:
                G.set_lora(lora)
                K = context_frames(m)
                ctx = load_tail(out, prev_tag)[-K:]
                guidance = it.get("cfg", cfg_cont)
                cands = []
                for c in range(max(1, n_cand)):
                    f_c, lat_c = G.context_clip(ctx, E, it["key"], it["seed"] + 100000 * c, guidance=guidance)
                    if anchor > 0:
                        if ref is None:
                            ref = lab_stats(to_uint8(read_video(f"{out}/videos/{shot_first}.mp4")))
                        f_c = np.concatenate([f_c[:K], colour_anchor(f_c[K:], ref, anchor)], axis=0)
                    cands.append((f_c, lat_c))
                sel = {}
                if len(cands) > 1:
                    first = read_video(f"{out}/videos/{shot_first}.mp4")
                    scores = [scorer.score(f_c, K, ctx, first, it.get("prompt", ""), shot_key=shot_first)
                              for f_c, _ in cands]
                    best = int(np.argmax([s["total"] for s in scores]))
                    for c, (f_c, _) in enumerate(cands):
                        if c != best:
                            export_to_video(list(f_c), f"{out}/videos/{tag}_rej{c}.mp4", fps=G.fps)
                    sel = {"chosen": best, "scores": scores}
                else:
                    best = 0
                frames, lat = cands[best]
            stats = {"sec": round(time.time() - t0, 1),
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
                     "frames_ok": frames_ok(frames)}
            if not new_shot:
                stats.update(cfg=guidance, **sel)
            save_clip(out, tag, frames, lat, G.fps)
        drop = 0 if new_shot else context_frames(m)
        clips.append(frames[drop:])
        pos += clips[-1].shape[0]
        prev_tag = tag
        rec = {"seq": name, "method": method, "m": m, "clip": i, "tag": tag, "cut": new_shot and i > 0,
               "cfg_cont": cfg_cont, "anchor": anchor, "lora": bool(lora) and not new_shot, **stats}
        log_jsonl(log_path, rec)
        print(rec, flush=True)
    G.set_lora(None)
    if not finalize:
        return None
    full = np.concatenate(clips, axis=0)
    export_to_video(list(full), f"{out}/videos/{name}_{method}_FULL.mp4", fps=G.fps)
    lengths = [int(c.shape[0]) for c in clips]
    json.dump({"seq": name, "method": method, "m": m, "cfg_cont": cfg_cont, "anchor": anchor, "lora": bool(lora),
               "clip_lengths": lengths, "boundaries": [int(b) for b in np.cumsum(lengths)[:-1]], "cuts": cuts,
               "prompts": [it.get("prompt", "") for it in items], "seconds": round(full.shape[0] / G.fps, 2)},
              open(f"{out}/manifests/{name}_{method}.json", "w"), indent=2)
    return full
