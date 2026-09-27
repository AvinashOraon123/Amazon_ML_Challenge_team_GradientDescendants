# Kaggle GPU job: prepare train data, train the bi-encoder, report held-out recall.
import glob, os, shutil, subprocess, sys, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii", "polars>=1.30"], check=True)

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input"
    return os.path.dirname(hits[0])

data_dir = find("train_source1.parquet")
code_dir = find("pipeline.py")
shutil.copytree(code_dir, "/kaggle/working/src", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
print("data:", data_dir, "code:", code_dir, flush=True)
subprocess.run(["nvidia-smi"])

def run(*args):
    t = time.time()
    subprocess.run([sys.executable, "pipeline.py", *args], check=True)
    print(f"== {args[0]} took {time.time() - t:.0f}s", flush=True)

WORK = "/tmp/work"   # large intermediates, not saved as output
run("prepare", "--data", data_dir, "--work", WORK, "--split", "train")
run("train-encoder", "--work", WORK, "--out", "/kaggle/working/models", "--epochs", "4", "--mine-from", "2")
run("block", "--work", WORK, "--split", "train", "--model", "/kaggle/working/models/encoder.pt", "--sweep")
shutil.copy(f"{WORK}/train/blocking_sweep.csv", "/kaggle/working/blocking_sweep.csv")
shutil.copy(f"{WORK}/train/prepare_stats.json", "/kaggle/working/prepare_stats.json")
