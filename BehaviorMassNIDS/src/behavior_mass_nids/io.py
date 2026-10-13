"""Portable capture cache. Labels are written/read only through separate files."""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .features import FEATURE_NAMES, FlowAccumulator
from .labels import PacketAttackLabelStream
from .pcap_reader import iter_packets

FIELDS = ("features", "timestamp_second", "source_node", "destination_node", "row_id")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_capture(path, normal=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text()) if path.is_dir() else {}
    if manifest.get("slice_seconds", 1) != 1:
        raise ValueError("This implementation requires one-second directional Flow slices")
    if normal and (manifest.get("source", {}).get("labels") is not None or manifest.get("evaluation_labels_attached", False) or (path / "labels.npz").exists()):
        raise ValueError("Normal training cannot load a capture with evaluation labels")
    if path.is_dir() and (path / "capture.npz").exists():
        with np.load(path / "capture.npz", allow_pickle=False) as f:
            data = {k: f[k] for k in FIELDS}
    elif path.is_dir():
        # Compatibility with the already evaluated graph-pilot cache. Ignore
        # all endpoint/connection identifiers except anonymous IP equality.
        with np.load(path / "observations.npz", allow_pickle=False) as f:
            data = {k: f[k] for k in ("features", "row_id", "timestamp_second")}
        with np.load(path / "structure.npz", allow_pickle=False) as f:
            data.update({k: f[k] for k in ("source_node", "destination_node")})
    else:
        if normal:
            raise ValueError("Normal training requires a directory with a capture manifest")
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in FIELDS}
    n = len(data["row_id"])
    if n == 0 or data["features"].shape != (n, 25):
        raise ValueError("Expected nonempty one-second Flow data with 25 statistics")
    np.testing.assert_array_equal(data["row_id"], np.arange(n))
    for k, v in data.items():
        if len(v) != n or not np.isfinite(v).all():
            raise ValueError(f"Invalid capture field {k}")
    if (data["features"] < 0).any() or (data["features"][:, 0] < 1).any():
        raise ValueError("Statistics must be nonnegative, with at least one packet")
    if np.any(np.diff(data["timestamp_second"]) < 0):
        raise ValueError("Flow rows must be ordered by integer capture second")
    for k in ("row_id", "timestamp_second", "source_node", "destination_node"):
        if not np.equal(data[k], np.floor(data[k])).all():
            raise ValueError(f"Expected integer {k}")
        data[k] = data[k].astype(np.int64)
    return data


def prepare_pcap(pcap, output, packet_labels=None):
    """Original directional tuple, 1-second bins, 2-second reorder tolerance.

    This is an offline extractor; raw endpoint strings exist only in transient
    aggregation keys and are replaced by anonymous equality indices on disk.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "capture.npz").exists() or (output / "labels.npz").exists():
        raise FileExistsError("Use a fresh capture output directory")
    active = defaultdict(dict)
    nodes = {}
    stats, seconds, source, destination, labels = [], [], [], [], []
    latest = flushed = None
    packets = 0
    stream = PacketAttackLabelStream(Path(packet_labels) if packet_labels else None)

    def node(ip):
        if ip not in nodes:
            nodes[ip] = len(nodes)
        return nodes[ip]

    def flush(t):
        for key, acc in sorted(active.pop(t).items()):
            stats.append(acc.features())
            seconds.append(t)
            source.append(node(key[0]))
            destination.append(node(key[1]))
            if packet_labels:
                labels.append(int(acc.attack))

    try:
        for packet in iter_packets(pcap, reader_path=Path(__file__).with_name("capture_reader.py"), allowed_protocols={"tcp", "udp"}):
            second = math.floor(packet.timestamp)
            if flushed is not None and second <= flushed:
                raise ValueError("Capture exceeded original two-second reorder tolerance")
            latest = second if latest is None else max(latest, second)
            for old in sorted(t for t in active if t < latest - 2):
                flush(old)
                flushed = old
            key = (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
            if key not in active[second]:
                active[second][key] = FlowAccumulator()
            active[second][key].add(packet, second, stream.is_attack(packet.frame_number))
            packets += 1
        for remaining in sorted(active):
            flush(remaining)
    finally:
        stream.close()
    if not stats:
        raise ValueError("No supported IPv4 TCP/UDP packets")
    ids = np.arange(len(stats), dtype=np.int64)
    np.savez_compressed(output / "capture.npz", row_id=ids, features=np.asarray(stats, np.float64),
        timestamp_second=np.asarray(seconds, np.int64), source_node=np.asarray(source, np.int64),
        destination_node=np.asarray(destination, np.int64))
    if packet_labels:
        np.savez_compressed(output / "labels.npz", row_id=ids, binary_label=np.asarray(labels, np.int8))
    write_json(output / "manifest.json", dict(flows=len(stats), packets=packets, slice_seconds=1,
        reorder_seconds=2, feature_names=FEATURE_NAMES, evaluation_labels_attached=bool(packet_labels),
        source={"pcap": Path(pcap).name, "labels": Path(packet_labels).name if packet_labels else None},
        length_policy="IPv4 ip.len clamped to 1500; not captured payload size",
        raw_identifiers_persisted=False, anonymous_nodes=len(nodes)))
