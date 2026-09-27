# Kaggle GPU job 3: error analysis with job-2 models; saves reusable train artifacts.
import glob, os, shutil, subprocess, sys, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anyascii", "polars>=1.30", "rapidfuzz>=3.6", "lightgbm>=4"], check=True)

def find(name, must_have=None):
    for h in sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)):
        d = os.path.dirname(h)
        if must_have is None or os.path.exists(os.path.join(d, must_have)):
            return d
    return None

data_dir = find("train_source1.parquet")
code_dir = find("pipeline.py", must_have="ber/run.py")
models_dir = find("policy.json", must_have="matcher/decision.json")
print("data:", data_dir, "code:", code_dir, "models:", models_dir, flush=True)
shutil.copytree(code_dir, "/kaggle/working/src", dirs_exist_ok=True)
shutil.copytree(models_dir, "/kaggle/working/models", dirs_exist_ok=True)
os.chdir("/kaggle/working/src")
subprocess.run([sys.executable, "pipeline.py", "analyze", "--data", data_dir, "--work", "/tmp/work",
                "--models", "/kaggle/working/models", "--out", "/kaggle/working/analysis"], check=True)
