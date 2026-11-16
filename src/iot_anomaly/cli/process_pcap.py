#!/usr/bin/env python3
"""
CLI: Extract session features from CTU-IoT-Malware PCAP files.

Usage
-----
iot-process-pcap \
    --pcap   data/raw/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap \
    --labels data/raw/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled \
    --output data/processed/ctu_features.csv

# Or process a whole directory of PCAP files (with a single label file):
iot-process-pcap \
    --pcap_dir data/raw/CTU-IoT-Malware-Capture-1-1/ \
    --labels   data/raw/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled \
    --output   data/processed/ctu_features.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

from iot_anomaly.data import PcapParser
from iot_anomaly.data.label_mapper import map_labels, parse_conn_log_labeled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("process_pcap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract session features from PCAP files")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap",     help="Path to a single .pcap file")
    group.add_argument("--pcap_dir", help="Directory containing .pcap files")
    p.add_argument("--labels", required=True,
                   help="Path to conn.log.labeled (NOT the bro/ directory)")
    p.add_argument("--output", required=True,
                   help="Output CSV path")
    p.add_argument("--session_timeout", type=float, default=120.0,
                   help="Seconds before a session is flushed (default 120)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    parser = PcapParser(session_timeout=args.session_timeout)

    if args.pcap:
        logger.info("Processing single PCAP: %s", args.pcap)
        df = parser.parse_file_with_labels(args.pcap, args.labels)
    else:
        logger.info("Processing PCAP directory: %s", args.pcap_dir)
        labeled_data = parse_conn_log_labeled(args.labels)
        raw_df = parser.parse_directory(args.pcap_dir)
        df = map_labels(raw_df, labeled_data)

    if df.empty:
        logger.error("No sessions extracted — check your PCAP and label paths.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d session rows to %s", len(df), args.output)
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
