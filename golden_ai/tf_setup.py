# -*- coding: utf-8 -*-
"""TensorFlow / Keras quiet setup (same idea as horizon/tf_setup.py)."""

from __future__ import annotations

import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

keras = tf.keras
