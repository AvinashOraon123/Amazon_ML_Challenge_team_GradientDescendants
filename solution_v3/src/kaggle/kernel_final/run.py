# Kaggle GPU job 6 (final): 50% entity holdout -> encoder trained on the other half, matcher (XGBoost on GPU)
# trained on 5x more unseen entities; all features (cleaned text, legal form, noise markers, name frequency,
# competition); resumable stages; searches saved as artifacts.
import glob, os, shutil, subprocess, sys, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BER_HOLDOUT_PCT"] = "50"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii", "polars>=1.30", "rapidfuzz>=3.6", "lightgbm>=4"], check=True)

def find(name, must_have=None, prefer=None):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(f"candidates for {name}: {hits}", flush=True)
    if prefer:
        hits = [h for h in hits if prefer in h] + [h for h in hits if prefer not in h]
    for h in hits:
        d = os.path.dirname(h)
        if must_have is None or os.path.exists(os.path.join(d, must_have)):
            return d
    return None

data_dir = find("train_source1.parquet")
code_dir = find("pipeline.py", must_have="ber/run.py", prefer="ber-code")
shutil.copytree(code_dir, "/kaggle/working/src", dirs_exist_ok=True)
os.makedirs("/kaggle/working/models", exist_ok=True)
prev = find("encoder_ckpt.pt", must_have="encoder_log.json", prefer="ber-final")   # resume a previous attempt
if prev:
    shutil.copytree(prev, "/kaggle/working/models", dirs_exist_ok=True)
    art = os.path.join(os.path.dirname(prev), "artifacts")
    for split in ("train", "test"):
        f = os.path.join(art, f"{split}_neighbours.npz")
        if os.path.exists(f):
            os.makedirs(f"/tmp/work/{split}", exist_ok=True)
            shutil.copy(f, f"/tmp/work/{split}/neighbours.npz")
            print("resumed", f, flush=True)
print("data:", data_dir, "code:", code_dir, "resume from:", prev, flush=True)
os.chdir("/kaggle/working/src")
subprocess.run(["nvidia-smi"])
import xgboost; print("xgboost", xgboost.__version__, flush=True)
t = time.time()
# never raise: Kaggle discards /kaggle/working when a job ends in an error, and the saved encoder,
# checkpoints and searches are what a resumed run needs
try:
    r = subprocess.run([sys.executable, "pipeline.py", "run-all", "--data", data_dir, "--work", "/tmp/work",
                        "--models", "/kaggle/working/models", "--out", "/kaggle/working/output"])
    print("PIPELINE EXIT CODE", r.returncode, flush=True)
    with open("/kaggle/working/PIPELINE_STATUS.txt", "w") as fh:
        print(f"exit {r.returncode}", file=fh)
finally:
    os.makedirs("/kaggle/working/artifacts", exist_ok=True)
    for split in ("train", "test"):
        f = f"/tmp/work/{split}/neighbours.npz"
        if os.path.exists(f):
            shutil.copy(f, f"/kaggle/working/artifacts/{split}_neighbours.npz")
    print(f"== run-all took {time.time() - t:.0f}s", flush=True)
