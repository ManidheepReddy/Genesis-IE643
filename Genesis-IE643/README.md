# Genesis — Long-form video by chaining short diffusion clips (IE 643, IIT Bombay)

Team **Genesis** · Moddu Manidheep Reddy (23b1311) · IE 643 Deep Learning: Theory and Practice, July–Dec 2026

Prompts (+ optional `[CUT]` lines) and a target duration go in; a long video made of chained
2-second Wan2.2 TI2V-5B clips comes out, together with automatic quality metrics.

## Methods
| Name | Clip *k+1* is generated from |
|---|---|
| `baseline` | text only (independent clips, hard cuts) |
| `A_lastframe` | last frame of clip *k* (m = 1) |
| `B_m2` / `B_m3` | last 5 / 9 frames of clip *k* → 2 / 3 clean latent frames (timestep 0, re-injected every step) |
| `C_anchor` | B + LAB colour anchoring to the shot's first clip (α = 0.7) + CFG 3.5 on continuation clips |
| `*_lora` | the same with the continuation LoRA (rank 32 on attention q/k/v/o, 47.2M params) |

## Repository layout
```
src/genesis_lib.py            chaining methods, context-latent conditioning, colour anchoring, [CUT] scripts
src/genesis_train_lora.py     OpenVid-1M preprocessing + continuation LoRA training (masked flow-matching loss)
src/genesis_queue_worker.py   two-GPU file-based job queue (dependencies, resume, time budget)
src/genesis_metrics.py        CLIP-T, prompt progress, DINO/CLIP consistency, seam jump, colour drift,
                              over-exposure, CLIP-IQA, optical-flow motion; tables and figures
src/genesis_app.py            Gradio interface
src/genesis_day6.py, genesis_day7.py   prompt planner, generate-and-select, per-join diagnosis, join marking
scripts/joins_analysis.py     novelty assessment: per-join diagnosis of the 20 s videos + colour hand-off fix
scripts/make_novelty_notebook.py   builds notebooks/Genesis_IE643_Novelty_Joins_and_Fix.ipynb
notebooks/                    the Kaggle notebooks actually run (Day 1–5 + interface) and the novelty notebooks
tests/                        CPU tests with a tiny random Wan2.2-shaped model (paths refer to our scratch layout)
results/                      metric tables of the evaluation runs (summary.md, per_video.csv, per_clip.csv)
results/novelty/              per-join tables and figures for the novelty assessment
```

## Running (Kaggle, free 2× T4)
1. Import a notebook from `notebooks/` (File → Import Notebook), set **GPU T4 x2** and **Internet on**.
2. Day 3 trains the LoRA and generates the evaluation set; attach its output as an input to later notebooks.
3. `Genesis_IE643_Interface.ipynb`: attach the Day 3 and Day 5 outputs, Run all, open the printed gradio.live link.

## Main dependencies
`diffusers` (Wan2.2 pipelines), `transformers`, `peft`, `torch`, `opencv-python`, `imageio[ffmpeg]`, `gradio`, `pandas`, `matplotlib`.
Model: [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Apache-2.0).
Training data: [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) (CC-BY-4.0), 300 filtered clips from part 0.

## Novelty assessment: what breaks at the joins
One 20 s fox story (11 clips, 10 joins, 480×832, same prompts and seed) generated with Method C and with C + LoRA (Day 5).
Every join is measured: pixel jump and colour jump (relative to an ordinary frame step) and optical-flow speed after/before.

| Run | Colour-jump joins | Median colour jump | Median pixel jump | Motion-break joins | ΔE last clip |
|---|---|---|---|---|---|
| C | 6/10 | 7.2 | 1.33 | 4/10 | 8.5 |
| C + LoRA | 7/10 | 3.9 | 1.24 | 4/10 | 8.0 |
| C + LoRA + colour hand-off | **0/10** | **0.5** | **1.08** | 4/10 | 8.0 |

The first recurring failure is a colour step at the join (from correcting each clip separately). The fix, a colour hand-off
(match the first frame after a join to the last frame before it, fade the correction out over 0.5 s), removes it while keeping
the long-term colour anchoring. Motion breaks remain and are the next target (m = 3 context).

Reproduce: `python scripts/joins_analysis.py --day5 <Day 5 output>/genesis5 --out results/novelty`
