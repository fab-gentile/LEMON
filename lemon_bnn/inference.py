"""Monte-Carlo dropout inference and uncertainty summary.

For each image, ``S`` stochastic forward passes give means ``mu_s`` and
log-variances ``log(sigma_s^2)`` for every parameter. Following the standard
decomposition for heteroscedastic Bayesian networks (Kendall & Gal 2017):

    prediction          = mean_s(mu_s)
    epistemic variance  = var_s(mu_s)
    aleatoric variance  = mean_s(exp(log sigma_s^2))
    total variance      = epistemic + aleatoric
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import keras
import numpy as np
import pandas as pd

from .config import Config
from .data import LensSequence, infer_input_shape
from .models import build_model
from .train import CONFIG_NAME, WEIGHTS_NAME
from .utils import set_seed


def load_trained_model(
    cfg: Config, weights_path: str, input_shape: Optional[Sequence[int]] = None
) -> keras.Model:
    """Rebuild the architecture from ``cfg`` and load weights saved by training."""
    shape = input_shape or cfg.input_shape
    if shape is None:
        raise ValueError("input_shape unknown: pass it or set cfg.input_shape.")
    model = build_model(
        tuple(shape), cfg.num_params, cfg.architecture, cfg.dropout_rate,
        cfg.l2_reg, mc_dropout=True, shift_param=cfg.shift_param,
    )
    model.load_weights(weights_path)
    return model


def mc_predict(
    model: keras.Model,
    x: np.ndarray,
    n_samples: int = 100,
    batch_size: int = 32,
    hdu: int = 0,
    data_root: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Run ``n_samples`` stochastic passes. Returns shape (N, 2P, n_samples).

    Each batch is read from disk once and passed through the network
    ``n_samples`` times.
    """
    seq = LensSequence(x, None, batch_size=batch_size, hdu=hdu, data_root=data_root)
    out_dim = model.output_shape[-1]
    samples = np.zeros((len(x), out_dim, n_samples), dtype=np.float32)
    start = 0
    for b in range(len(seq)):
        xb = seq[b]
        for s in range(n_samples):
            samples[start : start + len(xb), :, s] = model.predict_on_batch(xb)
        start += len(xb)
        if verbose and (b + 1) % max(1, len(seq) // 10) == 0:
            print(f"  MC inference: {b + 1}/{len(seq)} batches", flush=True)
    return samples


def summarize_samples(samples: np.ndarray, num_params: int) -> dict:
    """Collapse MC samples (N, 2P, S) into predictions and uncertainties.

    Returns arrays of shape (N, P): ``pred``, ``std_epistemic``,
    ``std_aleatoric`` and ``std_total``.
    """
    mu = samples[:, :num_params, :]
    log_var = samples[:, num_params:, :]
    pred = mu.mean(axis=-1)
    var_epi = mu.var(axis=-1)
    var_ale = np.exp(log_var).mean(axis=-1)
    return {
        "pred": pred,
        "std_epistemic": np.sqrt(var_epi),
        "std_aleatoric": np.sqrt(var_ale),
        "std_total": np.sqrt(var_epi + var_ale),
    }


def predictions_to_dataframe(
    samples: np.ndarray,
    param_names: Sequence[str],
    ids: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Tidy table: one row per lens, one column group per parameter."""
    n = samples.shape[0]
    summary = summarize_samples(samples, len(param_names))
    cols = {"id": np.arange(n) if ids is None else np.asarray(ids)}
    for j, name in enumerate(param_names):
        if y_true is not None:
            cols[f"{name}_true"] = y_true[:, j]
        cols[f"{name}_pred"] = summary["pred"][:, j]
        cols[f"{name}_std_epistemic"] = summary["std_epistemic"][:, j]
        cols[f"{name}_std_aleatoric"] = summary["std_aleatoric"][:, j]
        cols[f"{name}_std_total"] = summary["std_total"][:, j]
    return pd.DataFrame(cols)


def run_inference(
    x: np.ndarray,
    output_dir: str,
    ids: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    cfg: Optional[Config] = None,
    weights_path: Optional[str] = None,
    csv_name: str = "predictions.csv",
    samples_name: Optional[str] = None,
) -> pd.DataFrame:
    """Predict parameters and uncertainties for ``x`` and write a CSV.

    ``cfg`` and ``weights_path`` default to the files written by
    :func:`lemon_bnn.train.train_model` in ``output_dir``. If ``samples_name``
    is given, the raw MC samples (N, 2P, S) are also saved as ``.npy``.
    """
    cfg = cfg or Config.load(os.path.join(output_dir, CONFIG_NAME))
    weights_path = weights_path or os.path.join(output_dir, WEIGHTS_NAME)
    set_seed(cfg.seed)

    shape = cfg.input_shape or tuple(infer_input_shape(x, cfg.fits_hdu, cfg.data_root))
    model = load_trained_model(cfg, weights_path, shape)
    samples = mc_predict(
        model, x, cfg.n_mc_samples, cfg.predict_batch_size, cfg.fits_hdu, cfg.data_root
    )

    df = predictions_to_dataframe(samples, cfg.param_names, ids, y_true)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, csv_name), index=False)
    if samples_name:
        np.save(os.path.join(output_dir, samples_name), samples)
    return df
