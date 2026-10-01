"""Small helpers: seeding and Google Colab detection."""
from __future__ import annotations

import sys


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the Keras backend for reproducible runs."""
    import keras

    keras.utils.set_random_seed(seed)


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def mount_drive_if_colab(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive when running in Colab. Returns True if mounted."""
    if not in_colab():
        return False
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return True
