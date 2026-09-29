"""Streaming 1 s directional flow extraction using the repository packet reader."""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.flow_builder import PacketAttackLabelStream
from data.pcap_reader import PacketRecord, iter_packets
from . import FEATURE_NAMES

LOG = logging.getLogger(__name__)
IDENTITY = ("timestamp_second", "src_ip", "dst_ip", "src_port", "dst_port", "protocol")
READER = Path(__file__).resolve().parents[2] / "baselines/bpf-dag/data/pcap.py"


@dataclass
class FlowAccumulator:
    lengths: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    bin_bytes: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    bin_packets: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    attack: bool = False

    def add(self, packet: PacketRecord, second: int, attack: bool = False) -> None:
        self.lengths.append(packet.wire_length)
        self.times.append(packet.timestamp)
        micro = min(9, max(0, int((packet.timestamp - second) * 10)))
        self.bin_bytes[micro] += packet.wire_length
        self.bin_packets[micro] += 1
        self.attack |= attack

    def features(self) -> tuple[float, ...]:
        if len(self.lengths) == 1:
            length = float(self.lengths[0])
            return (1.0, length, 0.0, length, 0.0, length, length,
                    length, length, length, *(0.0,) * 8, 1.0,
                    length / 10, length * 0.3, length,
                    0.1, 0.3, 1.0)
        lengths = np.asarray(self.lengths, dtype=np.float64)
        times = np.sort(np.asarray(self.times, dtype=np.float64))
        iat = np.diff(times) if len(times) > 1 else np.empty(0)
        iat = np.maximum(iat, 0)
        if iat.size:
            iat_features = (float(iat.mean()), float(iat.std(ddof=0)),
                            float(iat.min()), float(iat.max()),
                            *map(float, np.percentile(iat, [25, 50, 75])),
                            float(iat.std(ddof=0) / iat.mean()) if iat.mean() > 0 else 0.0)
        else:
            iat_features = (0.0,) * 8
        result = (
            float(len(lengths)), float(lengths.sum()), float(times[-1] - times[0]),
            float(lengths.mean()), float(lengths.std(ddof=0)), float(lengths.min()),
            float(lengths.max()), *map(float, np.percentile(lengths, [25, 50, 75])),
            *iat_features, float(np.count_nonzero(self.bin_packets)),
            float(self.bin_bytes.mean()), float(self.bin_bytes.std(ddof=0)),
            float(self.bin_bytes.max()), float(self.bin_packets.mean()),
            float(self.bin_packets.std(ddof=0)), float(self.bin_packets.max()),
        )
        assert len(result) == len(FEATURE_NAMES) == 25
        return result


def extract(pcap: Path, out: Path, labels_out: Path | None = None,
            packet_labels: Path | None = None) -> dict[str, int]:
    """Read one packet at a time; retain accumulators for only the current second."""
    if packet_labels and not labels_out:
        raise ValueError("--packet-labels requires --labels-out")
    out.parent.mkdir(parents=True, exist_ok=True)
    if labels_out:
        labels_out.parent.mkdir(parents=True, exist_ok=True)
    label_stream = PacketAttackLabelStream(packet_labels)
    counts = {"packets": 0, "flows": 0, "attack_flows": 0}
    latest_second: int | None = None
    flushed_through: int | None = None
    # Capture merges can place a packet from the prior second after packets
    # from the next second. Keep three seconds in memory, never a full PCAP.
    active: dict[int, dict[tuple[str, str, int, int, str], FlowAccumulator]] = defaultdict(dict)
    label_handle = None
    try:
        with out.open("w", newline="") as feature_file:
            writer = csv.writer(feature_file)
            writer.writerow(["row_id", *IDENTITY, *FEATURE_NAMES])
            if labels_out:
                label_handle = labels_out.open("w", newline="")
                label_writer = csv.writer(label_handle)
                label_writer.writerow(["row_id", "binary_label"])

            def flush(second: int) -> None:
                for key, acc in sorted(active.pop(second).items()):
                    row_id = counts["flows"]
                    writer.writerow([row_id, second, key[0], key[1], key[2], key[3], key[4], *acc.features()])
                    if label_handle:
                        label_writer.writerow([row_id, int(acc.attack)])
                    counts["flows"] += 1
                    counts["attack_flows"] += int(acc.attack)
            for packet in iter_packets(pcap, reader_path=READER, allowed_protocols={"tcp", "udp"}):
                second = math.floor(packet.timestamp)
                if flushed_through is not None and second <= flushed_through:
                    raise ValueError(f"Capture timestamp is more than two seconds out of order at frame {packet.frame_number}")
                latest_second = second if latest_second is None else max(latest_second, second)
                ready = sorted(s for s in active if s < latest_second - 2)
                for old_second in ready:
                    flush(old_second)
                    flushed_through = old_second
                key = (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
                if key not in active[second]:
                    active[second][key] = FlowAccumulator()
                active[second][key].add(packet, second, label_stream.is_attack(packet.frame_number))
                counts["packets"] += 1
            for remaining_second in sorted(active):
                flush(remaining_second)
    finally:
        label_stream.close()
        if label_handle:
            label_handle.close()
    LOG.info("extracted %s: %s", pcap, counts)
    return counts
