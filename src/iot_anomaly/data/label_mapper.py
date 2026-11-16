"""
Label mapping utilities — joins Zeek/Bro conn.log.labeled ground-truth
labels onto a session-features DataFrame.

Fixed bugs vs original:
  BUG-10  Caller is responsible for supplying the correct *file* path
          (not a directory). A clear error is raised if given a directory.
  BUG-11  Label encoding uses case-insensitive matching so 'malicious'
          and 'Malicious' are both captured.
  BUG-12  DataFrame modifications use .loc to avoid SettingWithCopyWarning.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Protocol number → numeric string used as dict key
_PROTO_NUM: dict[str, str] = {"tcp": "6", "udp": "17", "icmp": "1"}


# ── conn.log.labeled parser ───────────────────────────────────────────────────

def parse_conn_log_labeled(file_path: str) -> dict[tuple, str]:
    """
    Parse a Zeek conn.log.labeled file and return a lookup dict.

    FIX BUG-10: raises a clear ValueError if *file_path* is a directory.
    Expected file path example:
        .../CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled

    Parameters
    ----------
    file_path : str
        Absolute path to the conn.log.labeled file (NOT the bro/ directory).

    Returns
    -------
    dict
        {(timestamp_str, src_ip, dst_ip, dst_port_str, proto_num_str): label}
    """
    if os.path.isdir(file_path):
        raise ValueError(
            f"Expected a file path but got a directory: {file_path}\n"
            "Hint: supply the path to 'conn.log.labeled', e.g.\n"
            "  .../bro/conn.log.labeled"
        )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label file not found: {file_path}")

    labels: dict[tuple, str] = {}
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 7:
                continue
            try:
                ts       = str(round(float(fields[0]), 6))
                src_ip   = fields[2]
                dst_ip   = fields[4]
                dst_port = fields[5]
                proto    = fields[6].strip().lower()
                label    = fields[-1].strip()
                proto_num = _PROTO_NUM.get(proto)
                if proto_num is None:
                    continue
                key = (ts, src_ip, dst_ip, dst_port, proto_num)
                labels[key] = label
            except Exception as exc:
                logger.debug("Skipping malformed line: %s — %s", line[:80], exc)
    logger.info("Loaded %d labeled sessions from %s", len(labels), file_path)
    return labels


# ── Label joining ──────────────────────────────────────────────────────────────

def _generate_key(row: pd.Series) -> tuple:
    ts = str(round(float(row["timestamp_session"]), 6))
    proto_num = (
        "6"  if row.get("TCP", 0) == 1
        else "17" if row.get("UDP", 0) == 1
        else "1"  if row.get("ICMP", 0) == 1
        else None
    )
    return (ts, row["src_ip"], row["dst_ip"], str(row["dst_port"]), str(proto_num))


def map_labels(df: pd.DataFrame, labeled_data: dict[tuple, str]) -> pd.DataFrame:
    """
    Join ground-truth labels onto *df*.

    FIX BUG-11: case-insensitive 'malicious' check.
    FIX BUG-12: uses .loc to avoid SettingWithCopyWarning.
    """
    df = df.copy()
    df.loc[:, "generated_key"] = df.apply(_generate_key, axis=1)
    df.loc[:, "label_raw"] = df["generated_key"].map(labeled_data).fillna("Unknown")

    # Normalise label strings
    df.loc[:, "label_raw"] = (
        df["label_raw"]
        .str.strip()
        .str.lower()
        .replace({"-   benign   -": "benign", "-   benign": "benign"})
    )

    pre_dist = df["label_raw"].value_counts()
    logger.info("Label distribution before filtering:\n%s", pre_dist.to_string())

    # Drop truly unknown rows (no match in labeled data)
    df = df[df["label_raw"] != "unknown"].copy()

    # Binary encoding — FIX BUG-11: lowercase 'malicious'
    df.loc[:, "label"] = df["label_raw"].apply(
        lambda x: 1 if "malicious" in x else 0
    )

    post_dist = df["label"].value_counts()
    logger.info("Binary label distribution:\n%s", post_dist.to_string())

    return df.drop(columns=["generated_key", "label_raw"])
