"""ResNet-style Bayesian CNN (Monte-Carlo dropout).

Every convolution is followed by spatial dropout. With ``mc_dropout=True`` the
dropout layers stay active at inference time, so repeated forward passes give
different predictions - the spread of which estimates the epistemic
uncertainty. The final dense layer outputs ``2 * num_params`` values:
``num_params`` means followed by ``num_params`` log-variances.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import keras
from keras import layers, regularizers

_BN_AXIS = -1  # channels_last


@keras.saving.register_keras_serializable(package="lemon_bnn")
class MCSpatialDropout2D(layers.SpatialDropout2D):
    """Spatial dropout that is *always* active, also at inference time.

    (Passing ``training=True`` when calling a standard dropout layer is not
    enough in Keras 3: the model-level ``training=False`` of predict()
    overrides it.)
    """

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


class _Dropout:
    """Factory for spatial dropout, optionally always active (MC dropout)."""

    def __init__(self, rate: float, mc_dropout: bool):
        self.rate, self.mc = rate, mc_dropout

    def __call__(self, x):
        cls = MCSpatialDropout2D if self.mc else layers.SpatialDropout2D
        return cls(self.rate)(x)


def _bn_relu(x):
    return layers.Activation("relu")(layers.BatchNormalization(axis=_BN_AXIS)(x))


def _conv(x, filters, kernel, strides, reg, drop):
    x = layers.Conv2D(
        filters,
        kernel,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(reg),
    )(x)
    return drop(x)


def _conv_bn_relu(x, filters, kernel, strides, reg, drop):
    """conv -> dropout -> BN -> ReLU (used for the stem)."""
    return _bn_relu(_conv(x, filters, kernel, strides, reg, drop))


def _bn_relu_conv(x, filters, kernel, strides, reg, drop):
    """BN -> ReLU -> conv -> dropout (pre-activation block, He et al. 2016)."""
    return _conv(_bn_relu(x), filters, kernel, strides, reg, drop)


def _shortcut(x, residual, reg):
    """Identity or 1x1-conv shortcut, merged with the residual by summation."""
    stride_h = int(round(x.shape[1] / residual.shape[1]))
    stride_w = int(round(x.shape[2] / residual.shape[2]))
    if stride_h > 1 or stride_w > 1 or x.shape[-1] != residual.shape[-1]:
        x = layers.Conv2D(
            residual.shape[-1],
            (1, 1),
            strides=(stride_h, stride_w),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(reg),
        )(x)
    return layers.add([x, residual])


def basic_block(x, filters, strides, first_of_first_layer, reg, drop):
    """Two 3x3 convolutions (ResNet-18/34)."""
    if first_of_first_layer:  # input already went through BN -> ReLU -> pool
        y = _conv(x, filters, (3, 3), strides, reg, drop)
    else:
        y = _bn_relu_conv(x, filters, (3, 3), strides, reg, drop)
    residual = _bn_relu_conv(y, filters, (3, 3), (1, 1), reg, drop)
    return _shortcut(x, residual, reg)


def bottleneck(x, filters, strides, first_of_first_layer, reg, drop):
    """1x1 -> 3x3 -> 1x1 bottleneck (ResNet-50)."""
    if first_of_first_layer:
        y = _conv(x, filters, (1, 1), strides, reg, drop)
    else:
        y = _bn_relu_conv(x, filters, (1, 1), strides, reg, drop)
    y = _bn_relu_conv(y, filters, (3, 3), (1, 1), reg, drop)
    residual = _bn_relu_conv(y, filters * 4, (1, 1), (1, 1), reg, drop)
    return _shortcut(x, residual, reg)


#: name -> (block function, blocks per stage)
ARCHITECTURES = {
    "resnet18": (basic_block, (2, 2, 2, 2)),
    "resnet34": (basic_block, (3, 4, 6, 3)),
    "resnet50": (bottleneck, (3, 4, 6, 3)),
}


def build_model(
    input_shape: Tuple[int, int, int],
    num_params: int,
    architecture: str = "resnet34",
    dropout_rate: float = 0.05,
    l2_reg: float = 2e-4,
    mc_dropout: bool = True,
    shift_param: Optional[Sequence[float]] = (-0.1, 0.1),
) -> keras.Model:
    """Build the network.

    Parameters
    ----------
    input_shape : (H, W, C) image shape; H and W should be >= 32.
    num_params : number of regressed parameters ``P``; output size is ``2 * P``.
    architecture : one of :data:`ARCHITECTURES`.
    mc_dropout : keep dropout active at inference (needed for uncertainties).
    shift_param : random-translation range as a fraction of the image size,
        applied during training only. ``None`` disables it.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Options: {list(ARCHITECTURES)}"
        )
    block_fn, repetitions = ARCHITECTURES[architecture]
    drop = _Dropout(dropout_rate, mc_dropout)

    inputs = keras.Input(shape=input_shape)
    x = inputs
    if shift_param is not None:
        x = layers.RandomTranslation(tuple(shift_param), tuple(shift_param))(x)
    x = _conv_bn_relu(x, 64, (7, 7), (2, 2), l2_reg, drop)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    filters = 64
    for stage, reps in enumerate(repetitions):
        for i in range(reps):
            strides = (2, 2) if (i == 0 and stage > 0) else (1, 1)
            x = block_fn(
                x,
                filters,
                strides,
                first_of_first_layer=(stage == 0 and i == 0),
                reg=l2_reg,
                drop=drop,
            )
        filters *= 2

    x = _bn_relu(x)
    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = drop(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(2 * num_params, kernel_initializer="he_normal")(x)
    return keras.Model(inputs, outputs, name=f"lemon_{architecture}")
