"""LeMoN / LEMON - Bayesian neural network for strong-lens parameter estimation.

Importing this package defaults the Keras backend to TensorFlow (override by
setting the ``KERAS_BACKEND`` environment variable *before* importing).
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from .config import Config  # noqa: E402
from .data import (  # noqa: E402
    LensSequence,
    infer_input_shape,
    load_dataset,
    preprocess_image,
    split_indices,
)
from .inference import (  # noqa: E402
    load_trained_model,
    mc_predict,
    predictions_to_dataframe,
    run_inference,
    summarize_samples,
)
from .losses import heteroscedastic_loss, rmse  # noqa: E402
from .models import ARCHITECTURES, build_model  # noqa: E402
from .train import train_model  # noqa: E402
from .utils import in_colab, mount_drive_if_colab, set_seed  # noqa: E402

__all__ = [
    "ARCHITECTURES",
    "Config",
    "LensSequence",
    "build_model",
    "heteroscedastic_loss",
    "in_colab",
    "infer_input_shape",
    "load_dataset",
    "load_trained_model",
    "mc_predict",
    "mount_drive_if_colab",
    "predictions_to_dataframe",
    "preprocess_image",
    "rmse",
    "run_inference",
    "set_seed",
    "split_indices",
    "summarize_samples",
    "train_model",
]
__version__ = "1.0.0"
