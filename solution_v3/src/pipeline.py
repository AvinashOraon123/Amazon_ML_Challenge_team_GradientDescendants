"""Command-line entry point. Run from this directory (src/).

    python pipeline.py prepare --data ../../../dataset --work ../work --split train [--sample 5]
    python pipeline.py train-encoder --work ../work --out ../models [--epochs 4]
    python pipeline.py block --work ../work --split train --model ../models/encoder.pt --sweep
    python pipeline.py run-all --data ../../../dataset --work ../work --models ../models --out ../../../output
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--sample", type=int, default=None, help="keep this percent of Source-1 entities (dev runs)")

    p = sub.add_parser("train-encoder")
    p.add_argument("--work", required=True, help="prepared train dir parent")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mine-from", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buckets-log2", type=int, default=22)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=200)

    p = sub.add_parser("block")
    p.add_argument("--work", required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sweep", action="store_true", help="evaluate candidate policies (train only)")

    p = sub.add_parser("run-all", help="whole pipeline, resumable; writes the two submission TSVs")
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mine-from", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buckets-log2", type=int, default=22)
    p.add_argument("--sample", type=int, default=None)

    p = sub.add_parser("analyze", help="train-side stages with saved models + error export")
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)

    a = ap.parse_args()
    if a.cmd == "prepare":
        from ber.prepare import prepare
        prepare(a.data, a.work, a.split, a.sample)
    elif a.cmd == "train-encoder":
        import torch
        from ber.train_encoder import train
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        train(Path(a.work) / "train", a.out, device=device, epochs=a.epochs, mine_from=a.mine_from,
              batch_size=a.batch_size, buckets_log2=a.buckets_log2, max_steps=a.max_steps, log_every=a.log_every)
    elif a.cmd == "block":
        import torch
        from ber import blocking
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        blocking.search(a.model, Path(a.work) / a.split, device, k=a.k)
        if a.sweep:
            blocking.sweep(Path(a.work) / a.split)
    elif a.cmd == "run-all":
        import torch
        from ber.run import run_all
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_all(a.data, a.work, a.models, a.out, device, epochs=a.epochs, mine_from=a.mine_from, sample=a.sample,
                batch_size=a.batch_size, buckets_log2=a.buckets_log2)
    elif a.cmd == "analyze":
        import torch
        from ber.run import analyze
        analyze(a.data, a.work, a.models, a.out, a.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
