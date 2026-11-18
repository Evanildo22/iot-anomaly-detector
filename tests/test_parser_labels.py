"""PCAP parsing and Zeek label joining (BUG-03/04/08/10/11/12 guards)."""

import warnings

import pandas as pd
import pytest

from iot_anomaly.data.label_mapper import map_labels, parse_conn_log_labeled
from iot_anomaly.data.parser import PcapParser, _session_key

from conftest import PCAP_PACKETS


# ── Parser ────────────────────────────────────────────────────────────────────

def test_session_key_is_bidirectional():
    a = _session_key("10.0.0.1", "10.0.0.2", 1234, 80, "TCP")
    b = _session_key("10.0.0.2", "10.0.0.1", 80, 1234, "TCP")
    assert a == b


def test_parse_file_groups_packets_into_sessions(pcap_file):
    rows = list(PcapParser(session_timeout=120).parse_file(pcap_file))
    df = pd.DataFrame(rows)
    # UDP pair, TCP pair, ICMP, late TCP → 4 sessions; the ARP frame is skipped.
    assert len(df) == 4
    assert sorted(df["packet_count"]) == [1, 1, 2, 2]
    assert set(df["protocol"]) == {"TCP", "UDP", "ICMP"}


def test_timestamps_keep_float_precision(pcap_file):
    # BUG-08: Decimal timestamps were replaced with float64.
    df = pd.DataFrame(PcapParser().parse_file(pcap_file))
    assert df["timestamp_session"].dtype == float
    assert 1001.0 in set(df["timestamp_session"])


def test_idle_sessions_are_flushed_before_end_of_file(pcap_file):
    # BUG-04: sessions are emitted after session_timeout seconds of inactivity,
    # so early sessions come out before the packet that arrives 300 s later.
    rows = list(PcapParser(session_timeout=120).parse_file(pcap_file))
    first_ts_of_late_packet = PCAP_PACKETS[-1][0]
    assert rows[-1]["timestamp_session"] == first_ts_of_late_packet
    assert all(r["timestamp_session"] < first_ts_of_late_packet for r in rows[:-1])


def test_payload_size_is_ip_payload_length(pcap_file):
    # BUG-03: payload_size is len(ip.data) — the IP header is not subtracted twice.
    df = pd.DataFrame(PcapParser().parse_file(pcap_file))
    udp = df[df["protocol"] == "UDP"].iloc[0]
    expected = [8 + len(b"query-a"), 8 + len(b"answer-a")]  # UDP header + data
    assert udp["payload_size_mean"] == pytest.approx(sum(expected) / 2)


def test_parse_directory(tmp_path, pcap_file):
    df = PcapParser().parse_directory(str(tmp_path))
    assert len(df) == 4


def test_parse_directory_without_pcaps_is_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert PcapParser().parse_directory(str(empty)).empty


# ── Label mapper ──────────────────────────────────────────────────────────────

def test_parse_conn_log_labeled(conn_log_file):
    labels = parse_conn_log_labeled(conn_log_file)
    assert len(labels) == 3  # igmp and malformed lines skipped
    assert labels[("1001.0", "192.168.1.10", "203.0.113.5", "23", "6")] == "Malicious   C&C"


def test_parse_conn_log_rejects_directory(tmp_path):
    # BUG-10: passing the bro/ directory used to fail obscurely.
    with pytest.raises(ValueError, match="directory"):
        parse_conn_log_labeled(str(tmp_path))


def test_parse_conn_log_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_conn_log_labeled(str(tmp_path / "missing.log"))


def test_map_labels_is_case_insensitive_and_drops_unknown(conn_log_file):
    df = pd.DataFrame([
        {"timestamp_session": 1000.0, "src_ip": "192.168.1.10", "dst_ip": "192.168.1.1",
         "dst_port": 53, "TCP": 0, "UDP": 1, "ICMP": 0},
        {"timestamp_session": 1001.0, "src_ip": "192.168.1.10", "dst_ip": "203.0.113.5",
         "dst_port": 23, "TCP": 1, "UDP": 0, "ICMP": 0},
        {"timestamp_session": 1002.0, "src_ip": "192.168.1.10", "dst_ip": "192.168.1.1",
         "dst_port": 0, "TCP": 0, "UDP": 0, "ICMP": 1},
        {"timestamp_session": 9999.0, "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2",
         "dst_port": 1, "TCP": 1, "UDP": 0, "ICMP": 0},
    ])
    with warnings.catch_warnings():
        # BUG-12: no SettingWithCopyWarning
        warnings.simplefilter("error", pd.errors.SettingWithCopyWarning)
        out = map_labels(df, parse_conn_log_labeled(conn_log_file))
    # BUG-11: 'Malicious' and 'malicious' both map to 1; unknown row dropped.
    assert out["label"].tolist() == [0, 1, 1]
    assert "generated_key" not in out and "label_raw" not in out
    assert len(df) == 4  # input not mutated


def test_parse_file_with_labels_end_to_end(pcap_file, conn_log_file):
    df = PcapParser().parse_file_with_labels(pcap_file, conn_log_file)
    assert sorted(df["label"].tolist()) == [0, 1, 1]
