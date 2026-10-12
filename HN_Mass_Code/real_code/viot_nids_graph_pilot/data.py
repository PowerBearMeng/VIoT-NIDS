"""Original Flow observations plus anonymous incidence indices; labels separate."""
from __future__ import annotations

import csv
import gzip
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.flow_builder import PacketAttackLabelStream
from data.pcap_reader import iter_packets
from viot_nids import FEATURE_NAMES
from viot_nids.features import FlowAccumulator, READER
from viot_nids_v7.data import source_signature, write_json
from . import SCHEMA_VERSION

LOG = logging.getLogger(__name__)
OBSERVATIONS = {"row_id", "timestamp_second", "features", "bin_packets", "bin_bytes", "first_bin"}
STRUCTURE = {"source_node", "destination_node", "source_endpoint", "destination_endpoint",
             "connection", "reverse_connection", "first_packet_time", "event_id"}


def prepare(pcap: Path, out: Path, labels: Path | None = None, reference: Path | None = None,
            diagnostic_observer=None, slice_seconds: int = 1):
    if not isinstance(slice_seconds, int) or slice_seconds < 1:
        raise ValueError('Slice length must be a positive integer number of seconds')
    signature = source_signature(pcap, labels)
    signature["graph_schema"] = SCHEMA_VERSION
    if slice_seconds != 1:
        signature['slice_seconds'] = slice_seconds
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["source"] != signature:
            raise ValueError(f"Source changed; use a new graph cache: {out}")
        load(out)
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    active = defaultdict(dict)
    frame_ids = defaultdict(dict)
    nodes, endpoints, connections = {}, {}, {}
    observations = {name: [] for name in OBSERVATIONS}
    structure = {name: [] for name in STRUCTURE}
    counts = dict(flows=0, packets=0, attack_flows=0)
    seconds = []
    latest = flushed = origin = None
    stream = PacketAttackLabelStream(labels)
    label_handle = gzip.open(out / "labels.csv.gz", "wt", newline="") if labels else None
    writer = csv.writer(label_handle) if label_handle else None
    if writer:
        writer.writerow(["row_id", "binary_label"])

    def ident(table, key):
        if key not in table:
            table[key] = len(table)
        return table[key]

    def flush(second):
        nonlocal origin
        if origin is None:
            origin = second
        members = sorted(active.pop(second).items())
        frames = frame_ids.pop(second)
        start = counts["flows"]
        for offset, (key, acc) in enumerate(members):
            row = start + offset
            observations["row_id"].append(row)
            observations["timestamp_second"].append(second)
            observations["features"].append(acc.features())
            observations["bin_packets"].append(acc.bin_packets)
            observations["bin_bytes"].append(acc.bin_bytes)
            first = min(acc.times)
            observations["first_bin"].append(min(9, max(0, int((first - second) * 10 / slice_seconds))))
            src, dst, sport, dport, protocol = key
            structure["source_node"].append(ident(nodes, src))
            structure["destination_node"].append(ident(nodes, dst))
            structure["source_endpoint"].append(ident(endpoints, (src, sport)))
            structure["destination_endpoint"].append(ident(endpoints, (dst, dport)))
            structure["connection"].append(ident(connections, key))
            structure["reverse_connection"].append(ident(connections, (dst, src, dport, sport, protocol)))
            structure["first_packet_time"].append(first - origin)
            structure["event_id"].append(frames[key])
            if writer:
                writer.writerow([row, int(acc.attack)])
            if diagnostic_observer is not None:
                diagnostic_observer(row, second, key, acc)
            counts["attack_flows"] += int(acc.attack)
        seconds.append(dict(second=second, rows=len(members), first_row=start))
        counts["flows"] += len(members)

    try:
        for packet in iter_packets(pcap, reader_path=READER, allowed_protocols={"tcp", "udp"}):
            packet_second = math.floor(packet.timestamp)
            second = math.floor(packet.timestamp / slice_seconds) * slice_seconds
            if flushed is not None and second <= flushed:
                raise ValueError(f"Timestamp disorder exceeds original buffer at frame {packet.frame_number}")
            latest = packet_second if latest is None else max(latest, packet_second)
            # Keep the original two seconds of packet reordering tolerance.
            # A longer Slice may only be flushed after its END plus the buffer.
            for old in sorted(t for t in active if t + slice_seconds <= latest - 2):
                flush(old)
                flushed = old
            key = (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
            if key not in active[second]:
                active[second][key] = FlowAccumulator()
                frame_ids[second][key] = packet.frame_number
            active[second][key].add(packet, second, stream.is_attack(packet.frame_number), slice_seconds)
            counts["packets"] += 1
            if counts["packets"] % 250000 == 0:
                LOG.info("Extract %s packets=%d flows=%d", pcap.parent.name, counts["packets"], counts["flows"])
        for second in sorted(active):
            flush(second)
    finally:
        stream.close()
        if label_handle:
            label_handle.close()
    if not counts["flows"]:
        raise ValueError("No supported Flows")
    obs = {k: np.asarray(v, dtype=np.float64 if k in {"features", "bin_packets", "bin_bytes"}
                         else np.int64) for k, v in observations.items()}
    st = {k: np.asarray(v, dtype=np.float64 if k == "first_packet_time" else np.int64)
          for k, v in structure.items()}
    if reference and slice_seconds == 1 and (reference / "manifest.json").exists():
        old = json.loads((reference / "manifest.json").read_text())
        if old["flows"] != counts["flows"]:
            raise ValueError("Normal Flow count changed")
        for shard in old["seconds"]:
            with np.load(reference / shard["file"], allow_pickle=False) as arrays:
                ix = arrays["row_id"]
                for k in OBSERVATIONS:
                    np.testing.assert_array_equal(obs[k][ix], arrays[k], err_msg=f"Original observation changed: {k}")
    np.savez_compressed(out / "observations.npz", **obs)
    np.savez_compressed(out / "structure.npz", **st)
    manifest = dict(schema=SCHEMA_VERSION, source=signature, **counts, seconds=seconds,
                    slice_seconds=slice_seconds, bin_seconds=slice_seconds / 10,
                    feature_names=list(FEATURE_NAMES), capture_origin=origin,
                    original_observation_alignment_verified=bool(reference),
                    anonymous_nodes=len(nodes), anonymous_endpoints=len(endpoints),
                    anonymous_connections_including_reverse=len(connections),
                    identity_policy="numeric indices for equality/incidence only; no raw identity mapping retained")
    write_json(manifest_file, manifest)
    LOG.info("Prepared %s: flows=%d packets=%d", out, counts["flows"], counts["packets"])
    return manifest


def load(cache: Path, normal=False):
    manifest = json.loads((cache / "manifest.json").read_text())
    if normal and manifest["source"]["labels"] is not None:
        raise ValueError("Training requires all-normal cache without labels")
    with np.load(cache / "observations.npz", allow_pickle=False) as f:
        if set(f.files) != OBSERVATIONS:
            raise ValueError("Unexpected observation fields")
        data = {k: f[k] for k in f.files}
    with np.load(cache / "structure.npz", allow_pickle=False) as f:
        if set(f.files) != STRUCTURE:
            raise ValueError("Unexpected structural fields")
        data.update({k: f[k] for k in f.files})
    n = manifest["flows"]
    if data["features"].shape != (n, 25):
        raise ValueError("Malformed Flow input")
    np.testing.assert_array_equal(data["row_id"], np.arange(n))
    if len(np.unique(data["event_id"])) != n:
        raise ValueError("Duplicate stable events")
    for key, value in data.items():
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"Malformed array {key}")
    np.testing.assert_array_equal(data["bin_packets"].sum(1), data["features"][:, 0])
    np.testing.assert_array_equal(data["bin_bytes"].sum(1), data["features"][:, 1])
    return data, manifest
