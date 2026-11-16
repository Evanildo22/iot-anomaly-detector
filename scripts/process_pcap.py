#!/usr/bin/env python3
"""Thin wrapper so `python scripts/process_pcap.py` works without installing.

The real implementation lives in iot_anomaly.cli.process_pcap (run
`iot-process-pcap` after `pip install -e .`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iot_anomaly.cli.process_pcap import main

if __name__ == "__main__":
    main()
