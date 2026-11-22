"""genesis_queue_worker.py — one process per GPU; pulls (sequence, method) jobs from a shared file queue.

A job runs once its dependencies are done ('base_<seq>' for chains, 'lora' for LoRA jobs). Jobs are claimed
atomically with O_EXCL lock files, so two workers never run the same job. Finished clips are skipped on re-run.
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


def st(name, kind):
    return os.path.exists(f"{Q}/{name}.{kind}")


def dep_state(d):
    if st(d, "done"):
        return "ok"
    if st(d, "failed") or st(d, "skipped"):
        return "dead"
    return "wait"


def next_job():
    """Returns (job, None) to run, (None, 'wait') if something may become runnable, (None, 'exit') otherwise."""
    waiting = False
    for job in plan["jobs"]:
        j = job["id"]
        if any(st(j, k) for k in ("done", "failed", "skipped", "claimed")):
            continue
        deps = [dep_state(d) for d in job.get("deps", [])]
        if "dead" in deps:
            open(f"{Q}/{j}.skipped", "w").write("dependency failed")
            continue
        if "wait" in deps:
            waiting = True
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
        gl.run_sequence(G, E, OUT, job["seq"], job["n_clips"], job["seeds"], job["method"], job["m"], log,
                        cfg_cont=job.get("cfg_cont"), anchor=job.get("anchor", 0.0), lora=lora)
        open(f"{Q}/{job['id']}.done", "w").write(a.worker)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        open(f"{Q}/{job['id']}.failed", "w").write(tb)
        G.set_lora(None)
        torch.cuda.empty_cache()
open(f"{OUT}/logs/worker{a.worker}.done", "w").write("ok")
print(f"worker {a.worker} done", flush=True)
