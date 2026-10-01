"""End-to-end smoke test on dummy data (no lens simulation).

Usage (from the repository root):
    python examples/smoke_test.py

Trains a tiny ResNet-18 for 2 epochs on 40 noise images, runs MC inference
with 5 samples and checks the output CSV. Takes about a minute on a CPU.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import lemon_bnn as lb  # noqa: E402
from make_example_data import make_example_data  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = make_example_data(os.path.join(tmp, "data"), n=40, size=64)
        out_dir = os.path.join(tmp, "out")

        cfg = lb.Config(
            architecture="resnet18", batch_size=8, epochs=2,
            n_mc_samples=5, predict_batch_size=8,
            data_root=data_dir, workers=0,
        )
        images, labels, ids = lb.load_dataset(
            os.path.join(data_dir, "images.npy"),
            os.path.join(data_dir, "labels.npy"),
            os.path.join(data_dir, "ids.npy"),
        )
        split = lb.split_indices(len(images), cfg.val_fraction, cfg.test_fraction, cfg.seed)
        tr, va, te = split["train"], split["val"], split["test"]

        # --- training --------------------------------------------------
        lb.train_model(cfg, images[tr], labels[tr], images[va], labels[va], out_dir, verbose=0)
        for f in ("weights.weights.h5", "config.json", "training_log.csv"):
            assert os.path.exists(os.path.join(out_dir, f)), f"missing {f}"

        # --- inference from FITS paths ---------------------------------
        df = lb.run_inference(
            images[te], out_dir, ids=ids[te], y_true=labels[te], samples_name="samples.npy"
        )
        assert len(df) == len(te)
        assert np.isfinite(df.drop(columns="id").to_numpy()).all(), "non-finite values"
        assert (df.filter(like="_std_") >= 0).all().all()
        assert os.path.exists(os.path.join(out_dir, "predictions.csv"))
        samples = np.load(os.path.join(out_dir, "samples.npy"))
        assert samples.shape == (len(te), 6, 5)
        # MC dropout must make passes differ
        assert samples[..., 0].std() > 0 and np.abs(samples[..., 0] - samples[..., 1]).max() > 0

        # --- inference from in-memory arrays (no FITS) ------------------
        arrays = np.random.default_rng(1).normal(size=(6, 64, 64)).astype("float32")
        cfg2 = lb.Config.load(os.path.join(out_dir, "config.json"))
        df2 = lb.run_inference(arrays, out_dir, cfg=cfg2, csv_name="pred_arrays.csv")
        assert len(df2) == 6

        # --- constant (completely empty) image must not give NaNs -------
        empty = np.zeros((2, 64, 64), dtype="float32")
        df3 = lb.run_inference(empty, out_dir, cfg=cfg2, csv_name="pred_empty.csv")
        assert np.isfinite(df3.drop(columns="id").to_numpy()).all()

        print(df.head(3).to_string())
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
