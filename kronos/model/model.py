"""
Kronos model loader and predictor wrapper.

Loads the Kronos time-series foundation model from HuggingFace Hub
and provides a clean :class:`KronosPredictor` interface for making
price predictions from OHLCV DataFrames.

Setup
-----
The Kronos model source must be available in your Python path.
The easiest way:

.. code:: bash

    pip install git+https://github.com/shiyu-coder/Kronos.git

Or, for an editable install:

.. code:: bash

    git clone https://github.com/shiyu-coder/Kronos.git
    pip install -e Kronos/

Once installed, ``from model.kronos import Kronos, KronosPredictor, KronosTokenizer``
will resolve.  This module wraps those imports with our configuration
layer (device detection, config values, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import get_logger

logger = get_logger(__name__)


class KronosUnavailableError(ImportError):
    """Raised when the upstream Kronos package cannot be imported.

    Install it with:

    .. code:: bash

        pip install git+https://github.com/shiyu-coder/Kronos.git
    """
    HELP = (
        "The upstream Kronos model package is required.\n"
        "  pip install git+https://github.com/shiyu-coder/Kronos.git\n"
        "See https://github.com/shiyu-coder/Kronos for details."
    )


# ---------------------------------------------------------------------------
# Lazy import gate -- attempt once at module level
# ---------------------------------------------------------------------------
_KRONOS_AVAILABLE = False
_Kronos: type = None  # type: ignore
_KronosTokenizer: type = None  # type: ignore
_KronosPredictor: type = None  # type: ignore

try:
    from model.kronos import Kronos as _KronosClass
    from model.kronos import KronosTokenizer as _KronosTokenizerClass
    from model.kronos import KronosPredictor as _KronosPredictorClass

    _Kronos = _KronosClass
    _KronosTokenizer = _KronosTokenizerClass
    _KronosPredictor = _KronosPredictorClass
    _KRONOS_AVAILABLE = True
    logger.info("Kronos model package loaded successfully")
except ImportError:
    logger.warning("Kronos model package not available. Install with: pip install git+https://github.com/shiyu-coder/Kronos.git")


def _check_kronos() -> None:
    """Raise :class:`KronosUnavailableError` if Kronos is not installed."""
    if not _KRONOS_AVAILABLE:
        raise KronosUnavailableError(KronosUnavailableError.HELP)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_kronos_model(
    config: KronosConfig,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Download and load the Kronos model from HuggingFace Hub.

    Parameters
    ----------
    config:
        Application configuration (``config.model_name``, ``config.resolved_device``
        are used when the explicit params are omitted).
    model_name:
        HuggingFace repo ID (default: ``config.model_name`` or
        ``"shiyu-coder/Kronos"``).
    device:
        Torch device string (default: ``config.resolved_device``).
    **kwargs:
        Additional arguments passed to ``Kronos.from_pretrained()``.

    Returns
    -------
    model
        The loaded Kronos model instance (``torch.nn.Module``).
    """
    _check_kronos()

    model_name = model_name or config.model_name
    device = device or config.resolved_device

    logger.info("Loading Kronos model: %s on %s", model_name, device)

    model: Any = _Kronos.from_pretrained(
        model_name,
        map_location=device,
        **kwargs,
    )
    model.eval()
    model.to(device)
    logger.info("Kronos model loaded successfully")
    return model


def load_kronos_tokenizer(config: KronosConfig) -> Any:
    """Load the Kronos tokenizer.

    The tokenizer is optional for prediction; if ``load_kronos_model``
    already handles it internally, this can be skipped.
    """
    _check_kronos()
    tokenizer: Any = _KronosTokenizer()
    return tokenizer


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------


class KronosPredictor:
    """High-level predictor wrapping Kronos model inference.

    Parameters
    ----------
    model:
        Loaded Kronos model (from :func:`load_kronos_model`).
    tokenizer:
        Kronos tokenizer (optional, can be ``None``).
    device:
        Torch device string.

    Example
    -------
    >>> config = KronosConfig()
    >>> model = load_kronos_model(config)
    >>> predictor = KronosPredictor(model, device=config.resolved_device)
    >>> pred = predictor.predict(df, x_timestamp=df.index[-1], pred_len=24)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        device: str = "cpu",
    ):
        _check_kronos()
        self._inner: Any = _KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        self.device = device

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: Any = None,
        y_timestamp: Any = None,
        pred_len: int = 24,
        T: float = 1.0,
        top_p: float = 0.95,
        sample_count: int = 20,
    ) -> Any:
        """Run Kronos prediction.

        Parameters
        ----------
        df:
            OHLCV DataFrame with columns ``['open', 'high', 'low', 'close']``
            and optionally ``'volume'``, ``'amount'``.  The index should be
            a ``DatetimeIndex`` with at most ``max_context`` rows.
        x_timestamp:
            The reference timestamp for the last known bar.  Typically
            ``df.index[-1]``.  If ``None``, inferred from the index.
        y_timestamp:
            Optional target timestamp for alignment.
        pred_len:
            Number of future candles to predict.
        T:
            Temperature for sampling (higher = more stochastic).
        top_p:
            Nucleus sampling threshold.
        sample_count:
            Number of sample trajectories to generate.

        Returns
        -------
        predictions
            The raw output from the upstream ``KronosPredictor.predict``.
        """
        return self._inner.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )

    def __repr__(self) -> str:
        return f"<KronosPredictor device={self.device}>"
