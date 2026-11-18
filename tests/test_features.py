"""Session feature extraction, including regression guards for BUG-01/02/05/06/07/09."""

import dpkt
import numpy as np
import pytest

from iot_anomaly.data import features as F
from iot_anomaly.data.features import compute_session_features

from conftest import make_packet

STAT_KEYS = ["min", "max", "median", "q1", "q3", "mean", "std"]


def test_packet_size_stats_values():
    stats = F.packet_size_stats(np.array([10.0, 20.0, 30.0, 40.0]))
    assert stats["packet_size_min"] == 10.0
    assert stats["packet_size_max"] == 40.0
    assert stats["packet_size_mean"] == 25.0
    assert stats["packet_size_median"] == 25.0
    assert stats["packet_size_q1"] == pytest.approx(17.5)
    assert stats["packet_size_std"] == pytest.approx(np.std([10, 20, 30, 40]))


def test_inter_arrival_keys_are_not_packet_size_keys():
    # BUG-01: IAT stats used to be returned under 'packet_size_*' keys.
    stats = F.inter_arrival_time_stats(np.array([0.0, 1.0, 3.0]))
    assert set(stats) == {f"inter_arrival_{k}" for k in STAT_KEYS}
    assert stats["inter_arrival_min"] == 1.0
    assert stats["inter_arrival_max"] == 2.0


def test_inter_arrival_sorts_timestamps():
    stats = F.inter_arrival_time_stats(np.array([3.0, 0.0, 1.0]))
    assert stats["inter_arrival_min"] == 1.0


@pytest.mark.parametrize("ts", [np.array([]), np.array([5.0])])
def test_inter_arrival_defaults_are_zero(ts):
    # BUG-07: defaults used to be 1 rather than 0.0.
    stats = F.inter_arrival_time_stats(ts)
    assert set(stats) == {f"inter_arrival_{k}" for k in STAT_KEYS}
    assert all(v == 0.0 for v in stats.values())


def test_empty_packet_stats_are_zero():
    assert all(v == 0.0 for v in F.packet_size_stats(np.array([])).values())


def test_packet_size_ratio():
    assert F.packet_size_ratio(np.array([50, 60, 200])) == 2.0
    assert F.packet_size_ratio(np.array([50, 60])) == 0.0  # no large packets


def test_protocol_distribution_uses_uppercase_keys():
    # BUG-05: lowercase keys never matched the parser's 'TCP'/'UDP'/'ICMP'.
    pkts = [make_packet(0, protocol="TCP"), make_packet(1, protocol="UDP"),
            make_packet(2, protocol="UDP"), make_packet(3, protocol="ICMP")]
    dist = F.protocol_distribution(pkts)
    assert dist == {"TCP": 0.25, "UDP": 0.5, "ICMP": 0.25}


def test_port_range_features_are_scalars():
    # BUG-06: port ranges used to be stored as tuples in a single column.
    pkts = [make_packet(0, src_port=1000, dst_port=80),
            make_packet(1, src_port=3000, dst_port=443)]
    ports = F.port_range_features(pkts)
    assert ports == {"src_port_min": 1000, "src_port_max": 3000,
                     "dst_port_min": 80, "dst_port_max": 443}
    assert all(isinstance(v, int) for v in ports.values())


def test_flag_combinations_return_raw_int_keys():
    # BUG-02: keys used to be prefixed twice ('flag_comb_flag_comb_…').
    ack = dpkt.tcp.TH_ACK
    pkts = [make_packet(0, tcp_flags=ack), make_packet(1, tcp_flags=ack),
            make_packet(2, tcp_flags=0)]
    combos = F.packet_flag_combinations(pkts)
    assert combos == {ack: pytest.approx(2 / 3), 0: pytest.approx(1 / 3)}


def test_packet_type_distribution():
    syn, ack = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK
    pkts = [make_packet(0, tcp_flags=syn), make_packet(1, tcp_flags=syn | ack)]
    dist = F.packet_type_distribution(pkts)
    assert dist["SYN"] == pytest.approx(2 / 3)
    assert dist["ACK"] == pytest.approx(1 / 3)
    assert dist["PSH"] == dist["URG"] == 0.0


def test_payload_byte_distribution_reads_payload_bytes():
    # BUG-09: payload bytes were never read, so these were always zero.
    dist = F.payload_byte_distribution([make_packet(0, payload=b"\x00\x00\x01")])
    assert dist["byte_sum"] == 3.0
    assert dist["byte_mean"] == pytest.approx(3 / 256)
    assert dist["byte_entropy"] > 0


def test_payload_byte_distribution_empty():
    dist = F.payload_byte_distribution([make_packet(0)])
    assert all(v == 0.0 for v in dist.values())


def test_sequence_number_anomaly_rate():
    pkts = [make_packet(i, tcp_seq=s) for i, s in enumerate([1, 2, 2, 3])]
    assert F.sequence_number_anomaly_rate(pkts) == 0.25


def test_ttl_distribution_has_256_bins_and_sums_to_one():
    pkts = [make_packet(0, ttl=64), make_packet(1, ttl=64), make_packet(2, ttl=128)]
    dist = F.ttl_distribution(pkts)
    assert len(dist) == 256
    assert dist["ttl_64"] == pytest.approx(2 / 3)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_direction_distribution(tcp_session):
    assert F.direction_distribution(tcp_session) == {"src_to_dst": 0.6, "dst_to_src": 0.4}


def test_compute_session_features(tcp_session):
    feats = compute_session_features(("sid",), tcp_session)
    assert feats["packet_count"] == 5
    assert feats["protocol"] == "TCP"
    assert feats["packet_size_max"] == 300.0
    assert feats["inter_arrival_max"] == 2.0
    assert feats["TCP"] == 1.0 and feats["UDP"] == 0.0
    assert feats["destination_ip_diversity"] == 2
    assert not any(k.startswith("flag_comb_flag_comb_") for k in feats)
    assert any(k.startswith("flag_comb_") for k in feats)
    assert len([k for k in feats if k.startswith("ttl_")]) == 256


def test_compute_session_features_single_packet():
    feats = compute_session_features(("sid",), [make_packet(1.0)])
    assert feats["inter_arrival_mean"] == 0.0
    assert feats["packet_interval_variance"] == 0.0
    assert feats["packet_timing_skewness"] == 0.0


def test_compute_session_features_empty_returns_none():
    assert compute_session_features(("sid",), []) is None
