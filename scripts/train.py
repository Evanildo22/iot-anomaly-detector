#!/usr/bin/env python3
"""Thin wrapper so `python scripts/train.py` works without installing.

The real implementation lives in iot_anomaly.cli.train (run `iot-train`
after `pip install -e .`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iot_anomaly.cli.train import main

if __name__ == "__main__":
    main()
