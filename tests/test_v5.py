"""Small deterministic V5 unit tests; no synthetic attack dataset or smoke run."""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

from data.pcap_reader import PacketRecord
from viot_nids import FEATURE_NAMES
from viot_nids.evaluation import metrics
from viot_nids.features import FlowAccumulator, extract
from viot_nids.neural import ChannelBalancedSampler, FlowMemory, channel_indices, embed
from viot_nids.pipeline import preprocessing
from viot_nids.statistics import assign_local, context_groups, responsibility


def packet(time: float, src: str = "a", dst: str = "b", src_port: int = 100,
           dst_port: int = 200, length: int = 100) -> PacketRecord:
    return PacketRecord(time, src, dst, src_port, dst_port, "udp", length)


def test_direction_boundary_features_and_microbins(monkeypatch, tmp_path: Path) -> None:
    packets = [packet(1.05), packet(1.15), packet(1.20, "b", "a", 200, 100),
               packet(2.00), packet(1.07)]
    monkeypatch.setattr("viot_nids.features.iter_packets", lambda *a, **k: iter(packets))
    out = tmp_path / "flows.csv"
    counts = extract(tmp_path / "not-read.pcap", out)
    assert counts["flows"] == 3
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["src_ip"] == "a" and rows[0]["dst_ip"] == "b"
    assert rows[1]["src_ip"] == "b" and rows[1]["dst_ip"] == "a"
    assert float(rows[0]["active_bin_count"]) == 2
    assert float(rows[0]["bin_packets_max"]) == 2
    assert float(rows[0]["active_duration"]) == 0.10 or np.isclose(float(rows[0]["active_duration"]), 0.10)
    assert float(rows[2]["iat_mean"]) == 0
    assert int(rows[2]["timestamp_second"]) == 2
    assert tuple(rows[0].keys())[-25:] == FEATURE_NAMES


def test_single_packet_iat_and_population_std() -> None:
    acc = FlowAccumulator()
    acc.add(packet(3.1), 3)
    x = acc.features()
    assert len(x) == 25
    assert x[10:18] == (0,) * 8
    assert x[4] == 0
    assert np.isclose(x[20], np.std([100] + [0] * 9, ddof=0))
    assert np.isclose(x[23], np.std([1] + [0] * 9, ddof=0))


def test_channel_continuity_sparse_delta_and_balancing() -> None:
    frame = pd.DataFrame({"src_ip": ["a", "a", "x"], "dst_ip": ["b", "b", "y"],
                          "protocol": ["udp"] * 3, "dst_port": [123, 123, 123],
                          "src_port": [1, 2, 9], "timestamp_second": [10, 70, 10]})
    channels = channel_indices(frame, ["src_ip", "dst_ip", "protocol", "dst_port"])
    assert sorted(map(len, channels)) == [1, 2]
    assert any(np.array_equal(c, [0, 1]) for c in channels)
    model = FlowMemory()
    captured = []
    original = model.step

    def observe(x, delta, h):
        captured.extend(delta.tolist())
        return original(x, delta, h)

    model.step = observe
    embed(model, np.zeros((3, 25), dtype=np.float32), frame.timestamp_second.to_numpy(), channels, "cpu")
    assert 60 in captured
    sampler = ChannelBalancedSampler([np.array([0]), np.arange(1, 1001)], seed=4)
    draws = sampler.draw(10000, 1)[:, 0]
    assert 0.47 < np.mean(draws == 0) < 0.53


def test_role_prior_excluded() -> None:
    a = GaussianMixture(n_components=1, covariance_type="diag", random_state=1).fit(
        np.array([[-1.1], [-1.0], [-0.9]]))
    b = GaussianMixture(n_components=1, covariance_type="diag", random_state=1).fit(
        np.array([[0.9], [1.0], [1.1]]))
    role, _, raw = assign_local(np.array([[1.0]]), {5: a, 9: b})
    assert role[0] == 9
    assert np.isclose(raw[0], -b.score_samples([[1.0]])[0])


def test_host_role_context_and_responsibility() -> None:
    frame = pd.DataFrame({"timestamp_second": [1, 1, 1, 2],
                          "src_ip": ["a", "a", "a", "a"],
                          "dst_ip": ["b", "b", "c", "b"],
                          "dst_port": [80, 80, 53, 80],
                          "byte_count": [100, 100, 20, 100],
                          "packet_count": [2, 2, 1, 2]})
    contexts, members, q = context_groups(frame, np.array([0, 0, 1, 0]))
    assert len(contexts) == 3
    assert contexts[0, 0] == 2 and contexts[0, 7] == 1 and contexts[0, 9] == 1
    assert np.allclose(q[:2, 3], 0.5)
    assert q[3, 5] == 0  # existing peer in same host/role history
    result = responsibility(np.array([[1., 100., 2., 1., 1., 1.], [1., 0., 0., 0., 0., 0.]]),
                            np.array([2, 100, 2, 50, 50, 1, 1, 1, 1, 1]),
                            np.zeros(6), np.ones(6))
    assert 0 <= result.min() <= result.max() <= 1
    assert result[0] > result[1]
    assert responsibility(np.zeros((1, 6)), np.zeros(10), np.zeros(6), np.ones(6))[0] == 0


def test_model_round_trip_and_metrics(tmp_path: Path) -> None:
    model = FlowMemory()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    second = FlowMemory()
    second.load_state_dict(torch.load(path, weights_only=True))
    x = torch.zeros(2, 25)
    dt = torch.tensor([0., 60.])
    h = torch.zeros(2, 32)
    assert torch.allclose(model.step(x, dt, h)[0], second.step(x, dt, h)[0])
    joblib.dump({"state": 1}, tmp_path / "stats.joblib")
    assert joblib.load(tmp_path / "stats.joblib")["state"] == 1
    result, roc, pr = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["AUROC"] == 1 and result["AUPRC"] == 1 and result["EER"] == 0
    assert {"fpr", "tpr"} <= set(roc) and {"precision", "recall"} <= set(pr)


def test_robust_scale_floor_is_persisted(tmp_path: Path) -> None:
    frame = pd.DataFrame(np.zeros((3, 25)), columns=FEATURE_NAMES)
    frame.loc[2, "iat_min"] = 0.000001
    _, scaler = preprocessing(frame, min_scale=0.05)
    assert scaler.scale_.min() >= 0.05
    path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, path)
    before, _ = preprocessing(frame, scaler)
    after, _ = preprocessing(frame, joblib.load(path))
    assert np.array_equal(before, after)


def test_long_channel_gru_matches_cell() -> None:
    torch.manual_seed(7)
    model = FlowMemory().eval()
    values = np.random.default_rng(7).normal(size=(1030, 25)).astype(np.float32)
    seconds = np.arange(1030, dtype=np.int64) // 10
    fast = embed(model, values, seconds, [np.arange(1030)], "cpu")
    hidden = torch.zeros(1, 32)
    slow = []
    with torch.no_grad():
        for i in range(len(values)):
            dt = 0 if i == 0 else int(seconds[i] - seconds[i - 1])
            z, hidden = model.step(torch.from_numpy(values[i:i + 1]), torch.tensor([dt], dtype=torch.float32), hidden)
            slow.append(z.numpy()[0])
    assert np.allclose(fast, slow, atol=2e-5)
