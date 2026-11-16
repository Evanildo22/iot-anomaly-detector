"""
Session-level feature extraction from raw packet lists.

Fixed bugs vs original:
  BUG-01  inter_arrival_time stats now return 'inter_arrival_*' keys (not 'packet_size_*').
  BUG-02  Flag-combination keys no longer double-prefixed.
  BUG-05  Protocol distribution uses uppercase keys matching dpkt class names.
  BUG-06  Port range split into separate _min / _max scalars.
  BUG-07  IAT defaults are 0.0, not 1.
  BUG-09  Payload byte distribution reads actual packet['payload'] bytes.
"""

from __future__ import annotations

import logging
from typing import Any

import dpkt
import numpy as np
from scipy.stats import entropy, skew
from sklearn.covariance import MinCovDet

logger = logging.getLogger(__name__)

# ── Packet statistics ──────────────────────────────────────────────────────────

def _packet_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """Generic descriptive statistics with a configurable key prefix."""
    if len(arr) == 0:
        keys = ["min", "max", "median", "q1", "q3", "mean", "std"]
        return {f"{prefix}_{k}": 0.0 for k in keys}
    return {
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q1":     float(np.percentile(arr, 25)),
        f"{prefix}_q3":     float(np.percentile(arr, 75)),
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
    }


def packet_size_stats(packet_sizes: np.ndarray) -> dict[str, float]:
    return _packet_stats(packet_sizes, "packet_size")


def inter_arrival_time_stats(timestamps: np.ndarray) -> dict[str, float]:
    """
    FIX BUG-01: returns 'inter_arrival_*' keys regardless of path taken.
    FIX BUG-07: defaults are 0.0, not 1.
    """
    if len(timestamps) < 2:
        keys = ["min", "max", "median", "q1", "q3", "mean", "std"]
        return {f"inter_arrival_{k}": 0.0 for k in keys}
    iat = np.diff(np.sort(timestamps))
    return _packet_stats(iat, "inter_arrival")


# ── Derived features ───────────────────────────────────────────────────────────

def packet_size_ratio(sizes: np.ndarray, threshold: int = 100) -> float:
    small = np.sum(sizes <= threshold)
    large = np.sum(sizes > threshold)
    return float(small / large) if large > 0 else 0.0


def packet_interval_variance(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    return float(np.var(np.diff(np.sort(timestamps))))


def destination_ip_diversity(packets: list[dict]) -> int:
    return len({p["dst_ip"] for p in packets})


def source_port_diversity(packets: list[dict]) -> int:
    return len({p["src_port"] for p in packets})


def packet_size_change_rate(sizes: np.ndarray) -> float:
    if len(sizes) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(sizes))))


def packet_timing_skewness(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    iat = np.diff(np.sort(timestamps))
    return float(skew(iat))


def direction_distribution(packets: list[dict]) -> dict[str, float]:
    if not packets:
        return {"src_to_dst": 0.0, "dst_to_src": 0.0}
    first_src = packets[0]["src_ip"]
    src_to_dst = sum(1 for p in packets if p["src_ip"] == first_src)
    total = len(packets)
    return {
        "src_to_dst": src_to_dst / total,
        "dst_to_src": (total - src_to_dst) / total,
    }


def payload_stats(packets: list[dict]) -> dict[str, float]:
    sizes = [p.get("payload_size", 0) for p in packets]
    if not sizes:
        return {"payload_size_mean": 0.0, "payload_size_variance": 0.0,
                "payload_size_magnitude": 0.0}
    arr = np.array(sizes, dtype=float)
    return {
        "payload_size_mean":      float(np.mean(arr)),
        "payload_size_variance":  float(np.var(arr)),
        "payload_size_magnitude": float(np.linalg.norm(arr)),
    }


def ttl_distribution(packets: list[dict]) -> dict[str, float]:
    """One-hot-ish TTL frequency distribution across all 256 values."""
    ttl_counts: dict[int, int] = {}
    for p in packets:
        ttl = p.get("ttl")
        if isinstance(ttl, int) and 0 < ttl < 256:
            ttl_counts[ttl] = ttl_counts.get(ttl, 0) + 1
    total = sum(ttl_counts.values()) or 1
    dist = {f"ttl_{i}": 0.0 for i in range(256)}
    for ttl, count in ttl_counts.items():
        dist[f"ttl_{ttl}"] = count / total
    return dist


def port_range_features(packets: list[dict]) -> dict[str, int]:
    """FIX BUG-06: split tuple into scalar min/max columns."""
    src_ports = [p["src_port"] for p in packets]
    dst_ports = [p["dst_port"] for p in packets]
    return {
        "src_port_min": int(min(src_ports)),
        "src_port_max": int(max(src_ports)),
        "dst_port_min": int(min(dst_ports)),
        "dst_port_max": int(max(dst_ports)),
    }


# ── TCP-specific features ──────────────────────────────────────────────────────

def packet_type_distribution(packets: list[dict]) -> dict[str, float]:
    counts = {"SYN": 0, "ACK": 0, "PSH": 0, "URG": 0}
    for p in packets:
        flags = p.get("tcp_flags", 0)
        if flags & dpkt.tcp.TH_SYN:  counts["SYN"] += 1
        if flags & dpkt.tcp.TH_ACK:  counts["ACK"] += 1
        if flags & dpkt.tcp.TH_PUSH: counts["PSH"] += 1
        if flags & dpkt.tcp.TH_URG:  counts["URG"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def packet_flag_combinations(packets: list[dict]) -> dict[str, float]:
    """
    FIX BUG-02: returns {int_flag: frequency} dicts — caller adds 'flag_comb_' prefix
    exactly once during DataFrame encoding.
    """
    combo_counts: dict[int, int] = {}
    for p in packets:
        flags = p.get("tcp_flags", 0)
        combo_counts[flags] = combo_counts.get(flags, 0) + 1
    total = sum(combo_counts.values()) or 1
    return {flags: count / total for flags, count in combo_counts.items()}


def sequence_number_anomaly_rate(packets: list[dict]) -> float:
    tcp_pkts = [p for p in packets if p.get("protocol") == "TCP" and "tcp_seq" in p]
    if len(tcp_pkts) < 2:
        return 0.0
    seqs = [p["tcp_seq"] for p in tcp_pkts]
    duplicates = len(seqs) - len(set(seqs))
    return duplicates / len(seqs)


def payload_byte_distribution(packets: list[dict]) -> dict[str, float]:
    """
    FIX BUG-09: reads packet['payload'] bytes (populated by parser).
    """
    payload_bytes = [b for p in packets for b in p.get("payload", b"")]
    if not payload_bytes:
        return {"byte_sum": 0.0, "byte_mean": 0.0,
                "byte_std": 0.0, "byte_entropy": 0.0}
    bc = np.bincount(payload_bytes, minlength=256).astype(float)
    return {
        "byte_sum":     float(bc.sum()),
        "byte_mean":    float(bc.mean()),
        "byte_std":     float(bc.std()),
        "byte_entropy": float(entropy(bc + 1e-10)),  # avoid log(0)
    }


def protocol_distribution(packets: list[dict]) -> dict[str, float]:
    """
    FIX BUG-05: counts uppercase protocol strings ('TCP', 'UDP', 'ICMP')
    matching the values stored by the parser.
    """
    protos = ["TCP", "UDP", "ICMP"]
    total = len(packets) or 1
    return {
        proto: sum(1 for p in packets if p["protocol"] == proto) / total
        for proto in protos
    }


# ── Session-level aggregation ─────────────────────────────────────────────────

def compute_session_features(session_id: tuple, packets: list[dict]) -> dict[str, Any] | None:
    """Aggregate all features for one network session."""
    if not packets:
        return None

    sizes = np.array([p["packet_size"] for p in packets], dtype=float)
    timestamps = np.array([p["timestamp"] for p in packets], dtype=float)
    first = packets[0]

    # Flag combinations — raw int keys; prefix added during DataFrame encoding
    flag_combos = packet_flag_combinations(packets)
    flag_combo_encoded = {f"flag_comb_{k}": v for k, v in flag_combos.items()}

    # Packet type distribution — individual columns
    ptd = packet_type_distribution(packets)

    features: dict[str, Any] = {
        # Identifiers (not fed to model, kept for debugging / label join)
        "session_id":          str(session_id),
        "timestamp_session":   float(first["timestamp"]),
        "src_ip":              first["src_ip"],
        "dst_ip":              first["dst_ip"],
        "src_port":            first["src_port"],
        "dst_port":            first["dst_port"],
        "protocol":            first["protocol"],
        "packet_count":        len(packets),
        # Packet size statistics
        **packet_size_stats(sizes),
        # Inter-arrival time statistics (FIX BUG-01)
        **inter_arrival_time_stats(timestamps),
        # Direction
        **direction_distribution(packets),
        # Payload
        **payload_stats(packets),
        # TTL distribution
        **ttl_distribution(packets),
        # Port ranges (FIX BUG-06)
        **port_range_features(packets),
        # Protocol distribution (FIX BUG-05)
        **protocol_distribution(packets),
        # Derived scalar features
        "packet_size_ratio":              packet_size_ratio(sizes),
        "packet_interval_variance":       packet_interval_variance(timestamps),
        "destination_ip_diversity":       destination_ip_diversity(packets),
        "source_port_diversity":          source_port_diversity(packets),
        "packet_size_change_rate":        packet_size_change_rate(sizes),
        "packet_timing_skewness":         packet_timing_skewness(timestamps),
        "packet_sequence_number_anomaly": sequence_number_anomaly_rate(packets),
        # TCP flag features (FIX BUG-02 — single prefix)
        **ptd,
        **flag_combo_encoded,
        # Payload byte distribution (FIX BUG-09)
        **payload_byte_distribution(packets),
    }
    return features
