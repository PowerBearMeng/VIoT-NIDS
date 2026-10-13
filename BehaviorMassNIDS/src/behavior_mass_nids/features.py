"""Original 25-statistic accumulator and IPv4 packet-length convention."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from .pcap_reader import PacketRecord
FEATURE_NAMES = (
    "packet_count", "byte_count", "active_duration",
    "packet_len_mean", "packet_len_std", "packet_len_min", "packet_len_max",
    "packet_len_p25", "packet_len_p50", "packet_len_p75",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_p25", "iat_p50",
    "iat_p75", "iat_cv", "active_bin_count", "bin_bytes_mean",
    "bin_bytes_std", "bin_bytes_max", "bin_packets_mean",
    "bin_packets_std", "bin_packets_max",
)

@dataclass
class FlowAccumulator:
    lengths: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    bin_bytes: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    bin_packets: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    attack: bool = False

    def add(self, packet: PacketRecord, second: int, attack: bool = False,
            slice_seconds: int = 1) -> None:
        self.lengths.append(packet.wire_length)
        self.times.append(packet.timestamp)
        micro = min(9, max(0, int((packet.timestamp - second) * 10 / slice_seconds)))
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
