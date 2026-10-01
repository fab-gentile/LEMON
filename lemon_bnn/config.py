"""Central configuration for training and inference."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass
class Config:
    """All user-tunable settings in one place.

    The defaults reproduce the setup of the original notebook (ResNet-34,
    dropout rate 0.05, Adam with lr=1e-4, batch size 64, 100 MC samples),
    with the three SIE parameters as targets.
    """

    # ---- targets -------------------------------------------------------
    #: Names of the regressed parameters (one network output pair each).
    param_names: Tuple[str, ...] = ("Ein_rad", "e1", "e2")

    # ---- data ----------------------------------------------------------
    #: Image shape (H, W, 1). ``None`` = inferred from the first image.
    input_shape: Optional[Tuple[int, int, int]] = None
    #: FITS extension to read.
    fits_hdu: int = 0
    #: Optional folder prepended to relative FITS paths.
    data_root: Optional[str] = None
    #: Fractions of the data held out for validation / testing.
    #: ``val_fraction=None`` reproduces the original notebook, where the
    #: test set doubled as the validation set.
    val_fraction: Optional[float] = 0.1
    test_fraction: float = 0.1

    # ---- model ---------------------------------------------------------
    architecture: str = "resnet34"  # resnet18 | resnet34 | resnet50
    dropout_rate: float = 0.05
    l2_reg: float = 2e-4
    #: Random-translation augmentation range (fraction of image size);
    #: ``None`` disables augmentation.
    shift_param: Optional[Tuple[float, float]] = (-0.1, 0.1)

    # ---- training ------------------------------------------------------
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-4
    lr_factor: float = 0.1
    lr_patience: int = 5
    min_lr: float = 1e-8
    min_delta: float = 1e-2
    early_stopping_patience: int = 10
    seed: int = 42
    #: Data-loading workers for the Keras ``PyDataset`` (0 = main thread).
    workers: int = 1

    # ---- inference -----------------------------------------------------
    n_mc_samples: int = 100
    predict_batch_size: int = 32

    # ------------------------------------------------------------------
    @property
    def num_params(self) -> int:
        return len(self.param_names)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        known = {f_.name for f_ in fields(cls)}
        raw = {k: v for k, v in raw.items() if k in known}
        for key in ("param_names", "input_shape", "shift_param"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)
