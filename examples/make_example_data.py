"""Create a tiny *dummy* dataset to check that the pipeline runs end to end.

The images are plain Gaussian noise and the labels are random numbers: there
is NO lens simulation here, so a network trained on this data learns nothing
meaningful. The only purpose is to exercise the code path (FITS reading,
training, MC inference, CSV output).

Writes into ``examples/data/``:
    fits/img_XXXX.fits   dummy FITS cutouts
    images.npy           (N,) array of FITS paths (relative to ``data_root``)
    labels.npy           (N, 3) random [Ein_rad, e1, e2]
    ids.npy              (N,) identifiers
"""
import argparse
import os

import numpy as np
from astropy.io import fits


def make_example_data(out_dir, n=40, size=64, seed=0):
    rng = np.random.default_rng(seed)
    fits_dir = os.path.join(out_dir, "fits")
    os.makedirs(fits_dir, exist_ok=True)

    rel_paths = []
    for i in range(n):
        name = f"img_{i:04d}.fits"
        img = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        fits.PrimaryHDU(img).writeto(os.path.join(fits_dir, name), overwrite=True)
        rel_paths.append(os.path.join("fits", name))

    labels = np.column_stack(
        [rng.uniform(0.5, 2.0, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)]
    ).astype(np.float32)

    np.save(os.path.join(out_dir, "images.npy"), np.array(rel_paths))
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "ids.npy"), np.arange(n))
    return out_dir


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(here, "data"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    print("Wrote dummy data to", make_example_data(args.out, args.n, args.size))
