"""Training loop."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import keras
import numpy as np

from .config import Config
from .data import LensSequence, infer_input_shape
from .losses import heteroscedastic_loss, rmse
from .models import build_model
from .utils import set_seed

WEIGHTS_NAME = "weights.weights.h5"  # Keras 3 requires the ".weights.h5" suffix
CONFIG_NAME = "config.json"
LOG_NAME = "training_log.csv"


def train_model(
    cfg: Config,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    output_dir: str,
    verbose: int = 1,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train the BNN and write weights, config and a CSV log to ``output_dir``.

    Returns the trained model (best weights restored) and the Keras history.
    """
    if y_train.shape[1] != cfg.num_params:
        raise ValueError(
            f"Labels have {y_train.shape[1]} columns but cfg.param_names has "
            f"{cfg.num_params} entries."
        )
    os.makedirs(output_dir, exist_ok=True)
    set_seed(cfg.seed)

    if cfg.input_shape is None:
        cfg.input_shape = tuple(infer_input_shape(x_train, cfg.fits_hdu, cfg.data_root))

    has_val = x_val is not None and len(x_val) > 0
    common = dict(
        batch_size=cfg.batch_size, hdu=cfg.fits_hdu, data_root=cfg.data_root,
        workers=cfg.workers, seed=cfg.seed,
    )
    train_seq = LensSequence(x_train, y_train, shuffle=True, **common)
    val_seq = LensSequence(x_val, y_val, shuffle=False, **common) if has_val else None
    monitor = "val_loss" if has_val else "loss"

    model = build_model(
        cfg.input_shape, cfg.num_params, cfg.architecture, cfg.dropout_rate,
        cfg.l2_reg, mc_dropout=True, shift_param=cfg.shift_param,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=heteroscedastic_loss,
        metrics=[rmse],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.min_lr, min_delta=cfg.min_delta, verbose=verbose,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=cfg.early_stopping_patience,
            min_delta=cfg.min_delta, restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(os.path.join(output_dir, LOG_NAME)),
    ]

    history = model.fit(
        train_seq, validation_data=val_seq, epochs=cfg.epochs,
        callbacks=callbacks, verbose=verbose,
    )

    model.save_weights(os.path.join(output_dir, WEIGHTS_NAME))
    cfg.save(os.path.join(output_dir, CONFIG_NAME))
    return model, history
