"""Shared fixtures: synthetic packets, PCAP files, label logs and tabular data.

Everything is generated at test time, so the suite needs no dataset
downloads and runs on a CI runner without a GPU.
"""

from __future__ import annotations

import os
import socket

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import dpkt
import numpy as np
import pandas as pd
import pytest


# ── Packet dictionaries (the parser's output format) ─────────────────────────

def make_packet(
    ts: float,
    size: int = 100,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    src_port: int = 40000,
    dst_port: int = 80,
    protocol: str = "TCP",
    ttl: int = 64,
    tcp_flags: int = 0,
    tcp_seq: int | None = None,
    payload: bytes = b"",
) -> dict:
    pkt = {
        "src_ip": src_ip, "dst_ip": dst_ip,
        "src_port": src_port, "dst_port": dst_port,
        "protocol": protocol, "timestamp": ts,
        "packet_size": size, "ttl": ttl,
        "tcp_flags": tcp_flags,
        "payload_size": len(payload), "payload": payload,
    }
    if tcp_seq is not None:
        pkt["tcp_seq"] = tcp_seq
    return pkt


@pytest.fixture
def tcp_session() -> list[dict]:
    """A five-packet bidirectional TCP session with known values."""
    syn, ack, psh = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH
    return [
        make_packet(0.0, 60, tcp_flags=syn, tcp_seq=1),
        make_packet(0.5, 60, src_ip="10.0.0.2", dst_ip="10.0.0.1",
                    src_port=80, dst_port=40000, tcp_flags=syn | ack, tcp_seq=7),
        make_packet(1.0, 200, tcp_flags=ack | psh, tcp_seq=2, payload=b"GET /"),
        make_packet(2.0, 300, src_ip="10.0.0.2", dst_ip="10.0.0.1",
                    src_port=80, dst_port=40000, tcp_flags=ack | psh, tcp_seq=8,
                    payload=b"HTTP/1.1 200 OK"),
        make_packet(4.0, 60, tcp_flags=ack, tcp_seq=2),
    ]


# ── PCAP and Zeek label files ────────────────────────────────────────────────

def _ethernet_frame(src: str, dst: str, transport, proto: int, ttl: int = 64) -> bytes:
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src), dst=socket.inet_aton(dst),
        p=proto, ttl=ttl, data=transport,
    )
    ip.len = len(ip)  # header + transport bytes
    eth = dpkt.ethernet.Ethernet(
        src=b"\x00\x11\x22\x33\x44\x55", dst=b"\x66\x77\x88\x99\xaa\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP, data=ip,
    )
    return bytes(eth)


# (timestamp, src, dst, sport, dport, proto, payload)
PCAP_PACKETS = [
    (1000.0, "192.168.1.10", "192.168.1.1", 50000, 53, "UDP", b"query-a"),
    (1000.1, "192.168.1.1", "192.168.1.10", 53, 50000, "UDP", b"answer-a"),
    (1001.0, "192.168.1.10", "203.0.113.5", 51000, 23, "TCP", b"login"),
    (1001.2, "203.0.113.5", "192.168.1.10", 23, 51000, "TCP", b"password:"),
    (1002.0, "192.168.1.10", "192.168.1.1", 0, 0, "ICMP", b"ping"),
    # Arrives > session_timeout after the UDP session, which forces an idle flush.
    (1300.0, "192.168.1.10", "198.51.100.7", 52000, 443, "TCP", b"hello"),
]


@pytest.fixture
def pcap_file(tmp_path) -> str:
    path = tmp_path / "capture.pcap"
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for ts, src, dst, sport, dport, proto, payload in PCAP_PACKETS:
            if proto == "TCP":
                transport = dpkt.tcp.TCP(sport=sport, dport=dport,
                                         flags=dpkt.tcp.TH_ACK, seq=1, data=payload)
                frame = _ethernet_frame(src, dst, transport, 6)
            elif proto == "UDP":
                transport = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
                transport.ulen = len(transport)
                frame = _ethernet_frame(src, dst, transport, 17)
            else:
                transport = dpkt.icmp.ICMP(type=8, code=0,
                                           data=dpkt.icmp.ICMP.Echo(id=1, seq=1, data=payload))
                frame = _ethernet_frame(src, dst, transport, 1)
            writer.writepkt(frame, ts=ts)
        # A non-IP frame (ARP) the parser must skip
        arp = dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_ARP, data=dpkt.arp.ARP())
        writer.writepkt(bytes(arp), ts=1002.5)
    return str(path)


@pytest.fixture
def conn_log_file(tmp_path) -> str:
    """Minimal Zeek conn.log.labeled with mixed-case labels (BUG-11)."""
    lines = [
        "#separator \\x09",
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tlabel",
        "1000.000000\tC1\t192.168.1.10\t50000\t192.168.1.1\t53\tudp\t-   benign   -",
        "1001.000000\tC2\t192.168.1.10\t51000\t203.0.113.5\t23\ttcp\tMalicious   C&C",
        "1002.000000\tC3\t192.168.1.10\t0\t192.168.1.1\t0\ticmp\tmalicious   PartOfAHorizontalPortScan",
        "1500.000000\tC4\t10.9.9.9\t1\t10.9.9.8\t2\tigmp\tBenign",  # unsupported proto
        "short\tline",
        "",
    ]
    path = tmp_path / "conn.log.labeled"
    path.write_text("\n".join(lines))
    return str(path)


# ── Tabular data resembling N-BaIoT rows ──────────────────────────────────────

def make_tabular(n: int = 800, n_features: int = 20, benign_share: float = 0.3,
                 seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Separable two-class data with some redundant, correlated columns."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > benign_share).astype(int)
    X = rng.normal(size=(n, n_features))
    X[:, :5] += y[:, None] * 2.5                      # informative columns
    X[:, 5] = X[:, 0] * 0.99 + rng.normal(scale=0.01, size=n)  # near-duplicate
    cols = [f"f{i:02d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


@pytest.fixture
def tabular():
    return make_tabular()
