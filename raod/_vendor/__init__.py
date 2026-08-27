"""Vendored third-party packages.

Adds this directory to ``sys.path`` so the vendored ``dinov3`` package can be
imported as a top-level module (its internal imports use absolute ``dinov3.*``
paths).
"""

import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
