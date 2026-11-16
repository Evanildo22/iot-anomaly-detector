"""
PCAP parser — session extraction from raw packet capture files.

Fixed bugs vs original:
  BUG-03  payload_size subtracted IP header twice; now uses len(ip.data) directly.
  BUG-04  Session chunking flushed live sessions; now uses idle-timeout flushing.
  BUG-08  Decimal timestamps replaced with float64.
  BUG-10  Caller must supply a *file* path, not a directory.
"""

from __future__ import annotations

import gc
import logging
import socket
import os
from typing import Iterator

import dpkt
import numpy as np
import pandas as pd

from .features import compute_session_features
from .label_mapper import parse_conn_log_labeled, map_labels

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_PORT = 0
DEFAULT_TTL = 64
MAX_WINDOW_SIZE = 65535

# Protocol number → friendly name (all uppercase to match dpkt class names)
PROTO_MAP: dict[int, str] = {6: "TCP", 17: "UDP", 1: "ICMP"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_valid_ip(ip_str: str) -> bool:
    try:
        socket.inet_aton(ip_str)
        return True
    except socket.error:
        return False


def _session_key(
    src_ip: str, dst_ip: str,
    src_port: int, dst_port: int,
    protocol: str,
) -> tuple:
    """Return a bidirectional session key (canonical order)."""
    a = (src_ip, dst_ip, src_port, dst_port, protocol)
    b = (dst_ip, src_ip, dst_port, src_port, protocol)
    return min(a, b)


# ── Packet info extractors ─────────────────────────────────────────────────────

def _extract_tcp_info(tcp: dpkt.tcp.TCP, payload_size: int) -> dict:
    tcp_flags = tcp.flags
    tcp_win = min(getattr(tcp, "win", 0), MAX_WINDOW_SIZE)
    if getattr(tcp, "win", 0) > MAX_WINDOW_SIZE:
        logger.warning("Clamped oversized TCP window: %d", tcp.win)
    return {
        "tcp_window_size": tcp_win,
        "tcp_flags": tcp_flags,
        "tcp_seq": getattr(tcp, "seq", 0),
        "payload_size": payload_size,
        "payload": bytes(tcp.data) if hasattr(tcp, "data") else b"",
    }


def _extract_udp_info(udp: dpkt.udp.UDP, payload_size: int) -> dict:
    return {
        "tcp_window_size": 0,
        "tcp_flags": 0,
        "payload_size": payload_size,
        "payload": bytes(udp.data) if hasattr(udp, "data") else b"",
    }


def _extract_icmp_info(icmp: dpkt.icmp.ICMP, payload_size: int) -> dict:
    return {
        "icmp_type": getattr(icmp, "type", 0),
        "icmp_code": getattr(icmp, "code", 0),
        "tcp_window_size": 0,
        "tcp_flags": 0,
        "payload_size": payload_size,
        "payload": b"",
    }


# ── Core parser ────────────────────────────────────────────────────────────────

class PcapParser:
    """
    Parse a PCAP file into per-session feature rows.

    Parameters
    ----------
    session_timeout : float
        Seconds of inactivity after which a session is considered complete
        and its features are emitted.  Prevents BUG-04.
    chunk_size : int
        Number of *completed* sessions to accumulate before appending to the
        result list (memory management).
    """

    def __init__(self, session_timeout: float = 120.0, chunk_size: int = 5_000):
        self.session_timeout = session_timeout
        self.chunk_size = chunk_size

    # ── public API ──────────────────────────────────────────────────────────

    def parse_file(self, pcap_path: str) -> Iterator[dict]:
        """Yield session-feature dicts from *pcap_path*."""
        logger.info("Parsing PCAP: %s", pcap_path)
        sessions: dict[tuple, list[dict]] = {}
        last_seen: dict[tuple, float] = {}

        try:
            with open(pcap_path, "rb") as fh:
                reader = dpkt.pcap.Reader(fh)
                for ts, buf in reader:
                    ts = float(ts)  # float64 — fixes BUG-08
                    packet = self._parse_packet(buf, ts)
                    if packet is None:
                        continue

                    sid = _session_key(
                        packet["src_ip"], packet["dst_ip"],
                        packet["src_port"], packet["dst_port"],
                        packet["protocol"],
                    )
                    sessions.setdefault(sid, []).append(packet)
                    last_seen[sid] = ts

                    # Flush idle sessions — fixes BUG-04
                    yield from self._flush_idle(sessions, last_seen, ts)

        except IOError as exc:
            logger.error("Cannot open %s: %s", pcap_path, exc)
            return

        # Flush remaining sessions
        for sid, pkts in sessions.items():
            feat = compute_session_features(sid, pkts)
            if feat:
                yield feat
        gc.collect()

    def parse_directory(self, directory: str) -> pd.DataFrame:
        """Parse all .pcap files in *directory* and return a combined DataFrame."""
        pcap_files = [
            os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if f.endswith(".pcap") and os.path.isfile(os.path.join(directory, f))
        ]
        if not pcap_files:
            logger.warning("No .pcap files found in %s", directory)
            return pd.DataFrame()

        rows = []
        for path in pcap_files:
            rows.extend(self.parse_file(path))
        return pd.DataFrame(rows)

    def parse_file_with_labels(
        self,
        pcap_path: str,
        labeled_log_path: str,
    ) -> pd.DataFrame:
        """Parse *pcap_path* and join ground-truth labels from *labeled_log_path*."""
        labeled_data = parse_conn_log_labeled(labeled_log_path)
        rows = list(self.parse_file(pcap_path))
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return map_labels(df, labeled_data)

    # ── internal helpers ────────────────────────────────────────────────────

    def _flush_idle(
        self,
        sessions: dict,
        last_seen: dict,
        current_ts: float,
    ) -> Iterator[dict]:
        idle_sids = [
            sid for sid, ts in last_seen.items()
            if current_ts - ts >= self.session_timeout
        ]
        for sid in idle_sids:
            feat = compute_session_features(sid, sessions.pop(sid))
            del last_seen[sid]
            if feat:
                yield feat

    def _parse_packet(self, buf: bytes, ts: float) -> dict | None:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except dpkt.UnpackError:
            return None

        if not isinstance(eth.data, dpkt.ip.IP):
            return None

        ip = eth.data
        src_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)

        if not _is_valid_ip(src_ip) or not _is_valid_ip(dst_ip):
            logger.debug("Invalid IP: %s → %s", src_ip, dst_ip)
            return None

        proto_num = ip.p
        protocol = PROTO_MAP.get(proto_num)
        if protocol is None:
            return None  # skip non TCP/UDP/ICMP

        packet_size = len(buf)
        ttl = getattr(ip, "ttl", DEFAULT_TTL)
        # FIX BUG-03: ip.data is already the IP payload; do not subtract header again.
        payload_size = len(ip.data)

        src_port = dst_port = DEFAULT_PORT
        extra: dict = {}

        if protocol == "TCP":
            tcp = ip.data
            src_port = getattr(tcp, "sport", DEFAULT_PORT)
            dst_port = getattr(tcp, "dport", DEFAULT_PORT)
            if not (0 <= src_port <= 65535 and 0 <= dst_port <= 65535):
                return None
            extra = _extract_tcp_info(tcp, payload_size)

        elif protocol == "UDP":
            udp = ip.data
            src_port = getattr(udp, "sport", DEFAULT_PORT)
            dst_port = getattr(udp, "dport", DEFAULT_PORT)
            if not (0 <= src_port <= 65535 and 0 <= dst_port <= 65535):
                return None
            extra = _extract_udp_info(udp, payload_size)

        elif protocol == "ICMP":
            extra = _extract_icmp_info(ip.data, payload_size)

        return {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "timestamp": ts,
            "packet_size": packet_size,
            "ttl": ttl,
            **extra,
        }
