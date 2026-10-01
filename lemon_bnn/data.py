"""Data loading, pre-processing and batching.

Expected inputs (all plain ``.npy`` files)
------------------------------------------
images : (N,) array of FITS file paths  *or*  (N, H, W[, 1]) array of images
labels : (N, P) array of target values (optional for inference-only runs)
ids    : (N,) array of identifiers (optional; defaults to 0..N-1)
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import keras
import numpy as np
from astropy.io import fits


# ----------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], then take the square root.

    Returns a float32 array of shape (H, W, 1). A constant (e.g. empty) image
    maps to zeros instead of producing NaNs.
    """
    img = np.nan_to_num(
        np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {img.shape}.")
    img = img - img.min()
    peak = img.max()
    if peak > 0:
        img = img / peak
    return np.sqrt(img)[..., None].astype(np.float32)


def _is_path_array(x: np.ndarray) -> bool:
    return x.dtype.kind in ("U", "S", "O")


def _read_source(item, hdu: int, data_root: Optional[str]) -> np.ndarray:
    """Read one image from a FITS path, or pass an in-memory array through."""
    if isinstance(item, (str, bytes, os.PathLike)):
        path = os.fsdecode(item)
        if data_root and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        return fits.getdata(path, ext=hdu, memmap=False)
    return np.asarray(item)


def infer_input_shape(
    x: np.ndarray, hdu: int = 0, data_root: Optional[str] = None
) -> Tuple[int, int, int]:
    """Shape (H, W, 1) of the images in ``x`` (read from the first one)."""
    if len(x) == 0:
        raise ValueError("Empty image array.")
    return preprocess_image(_read_source(x[0], hdu, data_root)).shape


# ----------------------------------------------------------------------
# Loading and splitting
# ----------------------------------------------------------------------
def load_dataset(
    images_path: str,
    labels_path: Optional[str] = None,
    ids_path: Optional[str] = None,
):
    """Load the ``.npy`` files. Returns ``(images, labels, ids)``.

    ``labels`` is ``None`` if no labels file is given; ``ids`` defaults to
    ``arange(N)``. NaNs in the labels are replaced by 0, as in the original
    notebook.
    """
    images = np.load(images_path, allow_pickle=True)
    labels = None
    if labels_path is not None:
        labels = np.nan_to_num(np.load(labels_path).astype(np.float32))
        if labels.ndim == 1:
            labels = labels[:, None]
        if len(labels) != len(images):
            raise ValueError(
                f"{len(images)} images but {len(labels)} label rows."
            )
    ids = np.load(ids_path, allow_pickle=True) if ids_path else np.arange(len(images))
    if len(ids) != len(images):
        raise ValueError(f"{len(images)} images but {len(ids)} ids.")
    return images, labels, ids


def split_indices(
    n: int,
    val_fraction: Optional[float] = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Random train/validation/test split.

    ``val_fraction=None`` reproduces the original notebook: the validation set
    *is* the test set (note that this leaks test information into early
    stopping and learning-rate scheduling).
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_fraction))
    test = perm[:n_test]
    if val_fraction is None:
        val = test
        train = perm[n_test:]
    else:
        n_val = int(round(n * val_fraction))
        val = perm[n_test : n_test + n_val]
        train = perm[n_test + n_val :]
    if len(train) == 0:
        raise ValueError("Training set is empty; reduce val/test fractions.")
    return {"train": train, "val": val, "test": test}


# ----------------------------------------------------------------------
# Batch generator
# ----------------------------------------------------------------------
class LensSequence(keras.utils.PyDataset):
    """Feeds pre-processed images (and labels) to Keras in batches.

    Images are read lazily, so datasets larger than memory are fine when
    ``x`` is an array of FITS paths. With ``y=None`` the sequence yields only
    images (inference).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 64,
        shuffle: bool = False,
        hdu: int = 0,
        data_root: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if y is not None and len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hdu, self.data_root = hdu, data_root
        self.indices = np.arange(len(x))
        self._rng = np.random.default_rng(seed)
        if shuffle:
            self._rng.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int):
        sel = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        xb = np.stack(
            [
                preprocess_image(_read_source(self.x[i], self.hdu, self.data_root))
                for i in sel
            ]
        )
        if self.y is None:
            return xb
        return xb, self.y[sel].astype(np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.indices)
