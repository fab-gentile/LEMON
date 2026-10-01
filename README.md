# LEMON – Lens MOdelling with Neural networks

LEMON is a **Bayesian neural network** that estimates the parameters of galaxy-scale strong gravitational lenses (by default the three parameters of a Singular Isothermal Ellipsoid: the **Einstein radius** and the **two ellipticity components**) directly from images, together with a reliable uncertainty for each prediction. It takes a fraction of a second per lens, which makes it suitable for the hundreds of thousands of lenses expected from Euclid and Rubin/LSST.

The code has been successfully employed so far in:
- Gentile et al. 2023, MNRAS, 522, 5442, [arXiv:2210.10793](https://arxiv.org/abs/2210.10793)
- Euclid Collaboration: Busillo et al. 2026, A&A 711, A31, [arXiv:2503.15329](https://arxiv.org/abs/2503.15329)

This repository contains the training and inference code. It is meant to be downloaded and used as is (no installation needed), either through the provided Jupyter notebook or by importing the modules in your own scripts.

> **Pretrained weights are not distributed with this repository.** They are available from the authors upon reasonable request (see [Contact](#contact)). You can also train your own network on your own data with the notebook.

---

## Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick check: does everything run?](#quick-check-does-everything-run)
- [Using your own data](#using-your-own-data)
- [Running the notebook](#running-the-notebook)
- [Using the modules in your own scripts](#using-the-modules-in-your-own-scripts)
- [Output format](#output-format)
- [Configuration reference](#configuration-reference)
- [Notes and caveats](#notes-and-caveats)
- [How to cite](#how-to-cite)
- [Contact](#contact)

---

## How it works

**Network.** A ResNet (18, 34 or 50 layers; ResNet-34 by default) with batch normalisation and L2 regularisation. A spatial-dropout layer follows every convolution. The final dense layer has `2P` outputs for `P` target parameters: the `P` predicted values `μ` and the `P` log-variances `log σ²`.

**Loss.** Heteroscedastic Gaussian negative log-likelihood, so the network learns the noise level of each prediction:

```
L = 0.5 · exp(−log σ²) · (y − μ)² + 0.5 · log σ²
```

**Bayesian inference (Monte-Carlo dropout).** Dropout stays **active at inference time**. Each image is passed `S` times (default 100) through the network, giving `S` different predictions `(μ_s, σ²_s)`. They are combined as:

| quantity | definition |
|---|---|
| prediction | mean over s of `μ_s` |
| epistemic uncertainty (model) | standard deviation over s of `μ_s` |
| aleatoric uncertainty (data) | square root of the mean over s of `σ²_s` |
| total uncertainty | square root of (epistemic² + aleatoric²) |

**Pre-processing.** Every image is shifted to a minimum of 0, divided by its maximum (so it lies in [0, 1]) and square-rooted. During training only, a random translation of up to ±10 % of the image size is applied as data augmentation.

---

## Repository structure

```
LeMoN/
├── lemon_bnn/                  # the library (import from the repository root)
│   ├── config.py               # Config dataclass: every setting in one place
│   ├── data.py                 # loading, pre-processing, batch generator, splitting
│   ├── models.py               # ResNet-18/34/50 Bayesian architectures
│   ├── losses.py               # heteroscedastic loss and RMSE metric
│   ├── train.py                # training
│   ├── inference.py            # MC-dropout inference and CSV output
│   └── utils.py                # seeding, Colab helpers
├── examples/
│   ├── make_example_data.py    # creates a tiny dummy dataset (noise, no lenses)
│   └── smoke_test.py           # end-to-end check that the code runs
├── LeMoN_train_and_predict.ipynb   # train + predict notebook (Colab and local)
├── requirements.txt
└── README.md
```

---

## Installation

Download or clone the repository; nothing needs to be installed beyond the dependencies.

```bash
git clone https://github.com/fab-gentile/LeMoN.git
cd LeMoN
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

Requirements: a recent Python 3 (tested with **Python 3.12**) and **TensorFlow ≥ 2.16 with Keras 3** (tested with TensorFlow 2.21 / Keras 3.15), `astropy`, `numpy`, `pandas`. A GPU is strongly recommended for training but not required.

**Google Colab:** nothing to install locally. Open `LeMoN_train_and_predict.ipynb` in Colab; the first cell clones the repository and installs the missing packages. Choose a GPU runtime (*Runtime → Change runtime type*) for training.

---

## Quick check: does everything run?

The repository ships a **dummy dataset generator** to verify that your installation works. The images are random noise and the labels are random numbers – there is **no lens simulation**, so the network learns nothing meaningful; the test only exercises the code (FITS reading, training, Monte-Carlo inference, CSV writing).

```bash
python examples/smoke_test.py
```

It trains a tiny ResNet-18 for 2 epochs on 40 noise images, runs 5 MC samples, checks the output files and finite values, and also checks prediction from in-memory arrays and from a completely empty (constant) image. It ends with `SMOKE TEST PASSED` (about a minute on a CPU).

You can also create the dummy files yourself to see the expected input format:

```bash
python examples/make_example_data.py     # writes examples/data/
```

The notebook does the same thing when `USE_EXAMPLE_DATA = True` (the default).

---

## Using your own data

Provide plain NumPy files:

| file | shape | content |
|---|---|---|
| `images.npy` | `(N,)` of strings **or** `(N, H, W)` / `(N, H, W, 1)` | paths to FITS files, **or** the images themselves |
| `labels.npy` | `(N, P)` | target values, one column per parameter (training/validation, optional for pure inference) |
| `ids.npy` | `(N,)` | optional identifiers, copied to the output CSV (default: `0 … N−1`) |

- **FITS files** are read from the extension given by `fits_hdu` (default 0). Relative paths are resolved against `data_root`. Reading is lazy, so datasets larger than memory are fine.
- **Image size** is free (inferred from the first image), but `H` and `W` should be at least 32 pixels because the network downsamples by a factor of 32. All images must have the same size.
- **Number of parameters** is free: set `param_names` in the config to as many names as there are label columns (e.g. add light-profile parameters). The network has `2P` outputs.
- NaNs in images and labels are replaced by 0.
- Images should be square cutouts centred on the lens, with consistent pixel scale and depth between training and prediction data. A network trained on one kind of data (e.g. simulated Euclid-like images) should not be expected to work on very different data (e.g. other instruments) without retraining.

---

## Running the notebook

Open `LeMoN_train_and_predict.ipynb` (Jupyter locally, or Colab) and:

1. Run the setup cell (it detects Colab/local automatically).
2. In the **configuration** cell, set `USE_EXAMPLE_DATA = False` and fill in `IMAGES_NPY`, `LABELS_NPY` (and optionally `IDS_NPY`, `DATA_ROOT`). Adjust the settings in `lb.Config(...)` if needed.
3. Run the remaining cells: load and split → train → predict.

Useful switches:

- `RUN_TRAINING = False` skips training and loads `weights.weights.h5` and `config.json` already present in `OUTPUT_DIR` – use this to run inference with weights you were given or trained before.
- `NEW_IMAGES_NPY = "..."` predicts on a separate dataset instead of the held-out test split (`NEW_LABELS_NPY` is optional and only adds `*_true` columns).
- `samples_name="mc_samples.npy"` in the inference cell also saves the raw MC samples of shape `(N, 2P, S)`.

Everything is written to `OUTPUT_DIR`:

| file | content |
|---|---|
| `weights.weights.h5` | trained weights (best epoch) |
| `config.json` | the exact configuration used (needed to rebuild the model for inference) |
| `training_log.csv` | loss and RMSE per epoch |
| `predictions.csv` | predictions and uncertainties (see below) |

---

## Using the modules in your own scripts

Run from the repository root (or add it to `sys.path`):

```python
import numpy as np
import lemon_bnn as lb

cfg = lb.Config(param_names=("Ein_rad", "e1", "e2"), architecture="resnet34", data_root="my_data/")

images, labels, ids = lb.load_dataset("images.npy", "labels.npy", "ids.npy")
split = lb.split_indices(len(images), cfg.val_fraction, cfg.test_fraction, cfg.seed)
tr, va, te = split["train"], split["val"], split["test"]

# Train (writes weights.weights.h5, config.json, training_log.csv to results/)
lb.train_model(cfg, images[tr], labels[tr], images[va], labels[va], "results/")

# Predict with uncertainties (writes results/predictions.csv)
df = lb.run_inference(images[te], "results/", ids=ids[te], y_true=labels[te])
```

Inference only, with weights you already have in `results/` (`weights.weights.h5` + `config.json`):

```python
new_images, _, new_ids = lb.load_dataset("new_images.npy")
df = lb.run_inference(new_images, "results/", ids=new_ids)
```

The lower-level functions `build_model`, `mc_predict`, `summarize_samples` and `predictions_to_dataframe` are also exported; see the docstrings.

---

## Output format

`predictions.csv` has one row per object and, for each parameter `<name>` in `param_names`:

| column | meaning |
|---|---|
| `id` | identifier from `ids.npy` |
| `<name>_true` | true value (only if labels were provided) |
| `<name>_pred` | prediction (mean over the MC samples) |
| `<name>_std_epistemic` | model (epistemic) uncertainty |
| `<name>_std_aleatoric` | data (aleatoric) uncertainty |
| `<name>_std_total` | total uncertainty, `sqrt(epistemic² + aleatoric²)` |

For the default parameters this gives the columns `Ein_rad_pred`, `Ein_rad_std_total`, `e1_pred`, `e1_std_total`, `e2_pred`, `e2_std_total`, and so on.

---

## Configuration reference

All settings live in `lemon_bnn.Config` (`lemon_bnn/config.py`); the defaults are those used for the published networks.

| setting | default | meaning |
|---|---|---|
| `param_names` | `("Ein_rad", "e1", "e2")` | names of the regressed parameters (one per label column) |
| `architecture` | `"resnet34"` | `resnet18`, `resnet34` or `resnet50` |
| `dropout_rate` | `0.05` | spatial-dropout rate (also sets the MC-dropout strength) |
| `l2_reg` | `2e-4` | L2 weight regularisation |
| `shift_param` | `(-0.1, 0.1)` | random-translation augmentation (fraction of image size); `None` disables it |
| `batch_size` | `64` | training batch size |
| `epochs` | `200` | maximum number of epochs |
| `learning_rate` | `1e-4` | initial Adam learning rate |
| `lr_factor`, `lr_patience`, `min_lr`, `min_delta` | `0.1`, `5`, `1e-8`, `1e-2` | ReduceLROnPlateau settings |
| `early_stopping_patience` | `10` | epochs without improvement before stopping (best weights restored) |
| `val_fraction`, `test_fraction` | `0.1`, `0.1` | held-out fractions; see the caveat below |
| `n_mc_samples` | `100` | stochastic forward passes per image at inference |
| `predict_batch_size` | `32` | inference batch size |
| `fits_hdu` | `0` | FITS extension to read |
| `data_root` | `None` | folder prepended to relative FITS paths |
| `workers` | `1` | data-loading threads (0 = main thread) |
| `seed` | `42` | random seed (split, initialisation, dropout) |

---

## Notes and caveats

- **Train / validation / test split.** By default the data are split into three disjoint sets (80/10/10 %). The earliest version of the code used the held-out 10 % both for validation (early stopping, learning-rate schedule) and for testing. To reproduce that behaviour, set `val_fraction=None`; be aware that it makes the test metrics slightly optimistic.
- **Dropout is part of the model.** The dropout layers are active during training, validation *and* inference. Always use the same `dropout_rate` for training and prediction; `config.json` takes care of this when you use the provided functions.
- **Weights format.** Weights are saved in the Keras 3 format (`*.weights.h5`). Weights produced by older TensorFlow/Keras 2 versions of this code are not directly loadable.
- **Reproducibility.** Seeds are fixed, but results on GPU can still differ slightly between runs and hardware.
- **Uncertainties** are only meaningful for images that resemble the training data.

---

## How to cite

If you use this code, please cite both papers:

```bibtex
@article{Gentile2023LeMoN,
  author  = {Gentile, Fabrizio and Tortora, Crescenzo and Covone, Giovanni and
             Koopmans, L{\'e}on V. E. and Li, Rui and Leuzzi, Laura and
             Napolitano, Nicola R.},
  title   = {{LeMoN: Lens Modelling with Neural networks -- I. Automated modelling
              of strong gravitational lenses with Bayesian Neural Networks}},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2023},
  doi     = {10.1093/mnras/stad1325},
  eprint  = {2210.10793},
  archivePrefix = {arXiv}
}

@article{Euclid2026LEMON,
  author  = {{Euclid Collaboration} and Busillo, V. and others},
  title   = {{Euclid Quick Data Release (Q1): XXXI. LEMON -- LEns MOdelling with
              Neural networks. Automated and fast modelling of Euclid
              gravitational lenses with singular isothermal ellipsoid mass profile}},
  journal = {Astronomy \& Astrophysics},
  volume  = {711},
  pages   = {A31},
  year    = {2026},
  eprint  = {2503.15329},
  archivePrefix = {arXiv}
}
```

- Gentile et al. 2023, MNRAS, 522, 5442, [arXiv:2210.10793](https://arxiv.org/abs/2210.10793)
- Euclid Collaboration: Busillo et al. 2026, A&A 711, A31, [arXiv:2503.15329](https://arxiv.org/abs/2503.15329)

---

## Contact

Questions, bug reports and requests for pretrained weights: please open a GitHub issue or contact the authors (see the papers above). Weights are provided upon reasonable request.
