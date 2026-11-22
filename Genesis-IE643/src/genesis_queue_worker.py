"""genesis_queue_worker.py (Day 6; Day 4 + generate-and-select and held-out jobs) — one process per GPU; pulls script jobs from a shared file queue.

Job = one script (list of clips, optionally split into shots by [CUT]) generated with one method.
A job runs once its dependencies are done. Jobs are claimed atomically with O_EXCL lock files.
Time budget: a job is only started if it can finish before config DEADLINE (epoch seconds), estimated as
(clips still to generate) x SEC_PER_CLIP; otherwise it is marked .skipped, so the commit never overruns.
"""
import argparse, json, os, sys, time, traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis_lib as gl

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--worker", required=True)
a = ap.parse_args()

plan = json.load(open(a.plan))
C = plan["config"]
OUT, Q = C["OUT"], f'{C["OUT"]}/queue'
os.makedirs(Q, exist_ok=True)
gl.SAVE_LATENTS = bool(C.get("SAVE_LATENTS", True))
DEADLINE, SEC_PER_CLIP = C.get("DEADLINE"), C.get("SEC_PER_CLIP", 480)


def st(name, kind):
    return os.path.exists(f"{Q}/{name}.{kind}")


def dep_state(d):
    if st(d, "done"):
        return "ok"
    if st(d, "failed") or st(d, "skipped"):
        return "dead"
    return "wait"


def clips_left(job):
    """Remaining work in units of one full-length clip (a 33-frame held-out clip costs ~0.6)."""
    if job.get("type") == "heldout":
        return 0.6 * sum(not os.path.exists(f"{OUT}/heldout/gen/{c['name']}__{x['id']}.mp4")
                         for c in job["conds"] for x in job["clips"])
    n = 0
    for i, it in enumerate(job["items"]):
        new_shot = i == 0 or it.get("cut")
        tag = f"{job['name']}_text_c{i}" if new_shot else f"{job['name']}_{job['method']}_c{i}"
        n += 0 if gl.clip_done(OUT, tag) else (1 if new_shot else job.get("n_cand", 1))
    return n


def next_job():
    """Returns (job, None) to run, (None, 'wait') if something may become runnable, (None, 'exit') otherwise."""
    waiting = False
    for job in plan["jobs"]:
        j = job["id"]
        if any(st(j, k) for k in ("done", "failed", "skipped", "claimed")):
            continue
        deps = [dep_state(d) for d in job.get("deps", [])]
        if "dead" in deps:
            open(f"{Q}/{j}.skipped", "w").write("dependency failed or skipped")
            continue
        if "wait" in deps:
            waiting = True
            continue
        if DEADLINE and time.time() + clips_left(job) * SEC_PER_CLIP > DEADLINE:
            open(f"{Q}/{j}.skipped", "w").write("time budget")
            print(f"worker {a.worker}: skip {j} (would overrun the time budget)", flush=True)
            continue
        try:
            fd = os.open(f"{Q}/{j}.claimed", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, a.worker.encode()); os.close(fd)
            return job, None
        except FileExistsError:
            continue
    return None, ("wait" if waiting else "exit")


G = gl.Genesis(C["WEIGHTS_DIR"], C["HEIGHT"], C["WIDTH"], C["NUM_FRAMES"], C["STEPS"], C["GUIDANCE"],
               C["FLOW_SHIFT"], C["FPS"], device=C.get("DEVICE", "cuda"), vae_offload=C.get("VAE_OFFLOAD", True),
               dtype=getattr(torch, C.get("DTYPE", "float16")))
open(f"{OUT}/logs/worker{a.worker}.ready", "w").write("ok")
print(f"worker {a.worker} loaded", flush=True)
E = torch.load(C["EMBEDS"])
log = f"{OUT}/logs/runs_worker{a.worker}.jsonl"
SCORER = None

while True:
    job, why = next_job()
    if job is None:
        if why == "exit":
            break
        time.sleep(60)
        continue
    print(f"worker {a.worker} -> {job['id']}", flush=True)
    try:
        lora = C["LORA_DIR"] if job.get("lora") else None
        if job.get("type") == "heldout":
            import genesis_day6 as g6
            g6.run_heldout(G, E, OUT, job["clips"], job["conds"], C["LORA_DIR"], log,
                           num_frames=job.get("num_frames", 33), cfg=job.get("cfg", 3.5))
        else:
            if job.get("n_cand", 1) > 1 and SCORER is None:
                import genesis_day6 as g6
                SCORER = g6.CandidateScorer(**C.get("SCORER_MODELS", {}))
            gl.run_script(G, E, OUT, job["name"], job["items"], job["method"], job["m"], log,
                          cfg_cont=job.get("cfg_cont"), anchor=job.get("anchor", 0.0), lora=lora,
                          finalize=job.get("finalize", True), n_cand=job.get("n_cand", 1), scorer=SCORER)
        open(f"{Q}/{job['id']}.done", "w").write(a.worker)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        open(f"{Q}/{job['id']}.failed", "w").write(tb)
        G.set_lora(None)
        torch.cuda.empty_cache()
open(f"{OUT}/logs/worker{a.worker}.done", "w").write("ok")
print(f"worker {a.worker} done", flush=True)
