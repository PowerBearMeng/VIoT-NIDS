"""Invariants for the new relational and regime-scoped evidence."""

import numpy as np
import pandas as pd

from viot_nids_v6.pipeline import _context, _relation_counts, _relation_raw


def test_relation_familiarity_accepts_service_port_at_either_endpoint():
    benign = pd.DataFrame([
        ("camera", "nvr", "tcp", 8554, 40000),
        ("camera", "nvr", "tcp", 8554, 40001),
        ("nvr", "camera", "tcp", 50000, 8080),
        ("nvr", "camera", "tcp", 50001, 8080),
    ], columns=["src_ip", "dst_ip", "protocol", "src_port", "dst_port"])
    candidate = pd.DataFrame([
        ("camera", "nvr", "tcp", 8554, 40100),
        ("nvr", "camera", "tcp", 50100, 8080),
        ("camera", "nvr", "tcp", 40100, 554),
        ("new_camera", "nvr", "tcp", 40100, 554),
    ], columns=benign.columns)
    raw, seen = _relation_raw(candidate, _relation_counts(benign))
    assert seen.tolist() == [2, 2, 0, 0]
    assert raw[0] == raw[1] == 0
    assert raw[2] > 0
    assert raw[3] > raw[2]  # Unknown pair backs off to global TCP port history.


def test_context_separates_regimes_within_one_host_and_second():
    frame = pd.DataFrame({
        "row_id": [0, 1, 2], "timestamp_second": [100, 100, 100],
        "src_ip": ["camera"] * 3, "dst_ip": ["nvr"] * 3,
        "dst_port": [554, 554, 8080],
        "byte_count": [60, 60, 80], "packet_count": [1, 1, 1],
    })
    contexts, members, codes = _context(frame, np.array([2, 2, 2]),
                                        np.array([3, 3, 1]))
    assert sorted(contexts[:, 0].tolist()) == [1, 2]
    assert sorted(codes.tolist()) == [33, 35]
    assert sorted(len(indices) for indices in members) == [1, 2]
