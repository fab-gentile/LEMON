"""Heteroscedastic Gaussian loss and metrics.

The network outputs ``2 * P`` numbers for ``P`` parameters: the first ``P`` are
the predicted values ``mu`` and the last ``P`` are the log-variances
``s = log(sigma^2)``. The loss is the Gaussian negative log-likelihood

    L = 0.5 * exp(-s) * (y - mu)^2 + 0.5 * s
"""
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="lemon_bnn")
def heteroscedastic_loss(y_true, y_pred):
    mu, log_var = ops.split(y_pred, 2, axis=-1)
    return 0.5 * ops.square(y_true - mu) * ops.exp(-log_var) + 0.5 * log_var


@keras.saving.register_keras_serializable(package="lemon_bnn")
def rmse(y_true, y_pred):
    """Root-mean-square error of the predicted values (log-variances ignored)."""
    mu, _ = ops.split(y_pred, 2, axis=-1)
    return ops.sqrt(ops.mean(ops.square(y_true - mu)))
