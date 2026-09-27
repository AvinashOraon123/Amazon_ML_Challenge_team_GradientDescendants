# Kaggle GPU job 2: full pipeline (resumable), reusing the encoder trained by job 1.
import glob, os, shutil, subprocess, sys, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii", "polars>=1.30", "rapidfuzz>=3.6", "lightgbm>=4"], check=True)

def find(name, must_have=None):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(f"candidates for {name}: {hits}", flush=True)
    for h in hits:
        d = os.path.dirname(h)
        if must_have is None or os.path.exists(os.path.join(d, must_have)):
            return d
    return None

data_dir = find("train_source1.parquet")
code_dir = find("pipeline.py", must_have="ber/run.py")   # the ber-code dataset, not job 1's old copy
enc_dir = find("encoder.pt", must_have="encoder_log.json")
shutil.copytree(code_dir, "/kaggle/working/src", dirs_exist_ok=True)
os.makedirs("/kaggle/working/models", exist_ok=True)
if enc_dir:   # reuse job-1 encoder (and its log / checkpoint) -> encoder stage is skipped or resumed
    for f in ("encoder.pt", "encoder_log.json"):     # encoder only: policy + matcher are recomputed
        if os.path.isfile(os.path.join(enc_dir, f)):
            shutil.copy(os.path.join(enc_dir, f), "/kaggle/working/models/")
# reuse saved train neighbours (job 3 artifact) -> skips the ~50 min train search; features are rebuilt
art = find("neighbours.npz", must_have="features.parquet")
if art:
    os.makedirs("/tmp/work/train", exist_ok=True)
    shutil.copy(os.path.join(art, "neighbours.npz"), "/tmp/work/train/neighbours.npz")
print("data:", data_dir, "code:", code_dir, "encoder:", enc_dir, "artifact:", art, os.listdir("/kaggle/working/models"), flush=True)
os.chdir("/kaggle/working/src")
subprocess.run(["nvidia-smi"])
t = time.time()
subprocess.run([sys.executable, "pipeline.py", "run-all", "--data", data_dir, "--work", "/tmp/work",
                "--models", "/kaggle/working/models", "--out", "/kaggle/working/output"], check=True)
print(f"== run-all took {time.time() - t:.0f}s", flush=True)
