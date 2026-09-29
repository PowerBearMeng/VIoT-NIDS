"""Build configurable fixed-window logical flow segments."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .microbin_features import (
    DirectionalIATAccumulator,
    MicroBinAccumulator,
    make_accumulator,
)
from .pcap_reader import PacketRecord, iter_packets


@dataclass(frozen=True)
class CaptureSpec:
    path: Path
    label: str
    split: str
    packet_labels: Path | None = None
    dataset_name: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass(frozen=True)
class SegmentRecord:
    features: np.ndarray
    segment_id: str
    flow_id: str
    capture: str
    capture_id: str
    split: str
    label_name: str
    endpoint_a_ip: str
    endpoint_b_ip: str
    endpoint_a_port: int
    endpoint_b_port: int
    protocol: str
    window_index: int
    segment_start: float
    packet_count: int
    byte_count: int
    dataset_name: str


class PacketAttackLabelStream:
    """Sequentially align Gotham or official Kitsune labels to PCAP frames."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle = None
        self.reader = None
        self.pending_row: list[str] | None = None
        self.frame_column = -1
        self.label_column = -1
        self.sequential_frame = 0
        self.current_frame = 0
        self.current_label = False
        if path is not None:
            self.handle = path.open(
                "r", encoding="utf-8-sig", errors="strict", newline=""
            )
            self.reader = csv.reader(self.handle)
            try:
                first = next(self.reader)
            except StopIteration as error:
                self.handle.close()
                raise ValueError(f"Packet label CSV {path} is empty") from error
            header = [value.strip().lower() for value in first]
            if "frame_number" in header and "binary_label" in header:
                self.frame_column = header.index("frame_number")
                self.label_column = header.index("binary_label")
            elif len(header) >= 2 and header[-1] == "x":
                # Official Kitsune R export: empty/index column followed by x.
                self.frame_column = 0
                self.label_column = len(header) - 1
            elif len(first) == 1 and first[0].strip() in {"0", "1"}:
                # Official Mirai vector: headerless sequential 0/1 rows.
                self.frame_column = -1
                self.label_column = 0
                self.pending_row = first
            else:
                self.handle.close()
                raise ValueError(
                    f"Packet label CSV {path} must be Gotham frame_number/binary_label, "
                    "an official Kitsune index/x export, or a headerless 0/1 vector"
                )

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()

    def is_attack(self, frame_number: int) -> bool:
        if self.handle is None:
            return False
        while self.current_frame < frame_number:
            assert self.reader is not None
            if self.pending_row is not None:
                columns = self.pending_row
                self.pending_row = None
            else:
                try:
                    columns = next(self.reader)
                except StopIteration:
                    columns = []
            if not columns:
                raise ValueError(
                    f"Packet label CSV {self.path} ended before PCAP frame {frame_number}"
                )
            required = self.label_column if self.frame_column < 0 else max(
                self.frame_column, self.label_column
            )
            if len(columns) <= required:
                raise ValueError(f"Malformed packet label row in {self.path}")
            if self.frame_column < 0:
                self.sequential_frame += 1
                self.current_frame = self.sequential_frame
            else:
                self.current_frame = int(columns[self.frame_column])
            label = int(float(columns[self.label_column].strip()))
            if label not in {0, 1}:
                raise ValueError(f"Invalid packet label {label} in {self.path}")
            self.current_label = label == 1
        if self.current_frame != frame_number:
            raise ValueError(
                f"Packet labels in {self.path} skipped PCAP frame {frame_number}"
            )
        return self.current_label


def _endpoint_key(ip: str, port: int) -> tuple[int, bytes, int]:
    address = ipaddress.ip_address(ip)
    return address.version, address.packed, int(port)


def canonical_flow(packet: PacketRecord) -> tuple[tuple[object, ...], int]:
    """Return canonical bidirectional key and 0=A->B / 1=B->A direction."""
    src = (packet.src_ip, int(packet.src_port))
    dst = (packet.dst_ip, int(packet.dst_port))
    if _endpoint_key(*src) <= _endpoint_key(*dst):
        a, b, direction = src, dst, 0
    else:
        a, b, direction = dst, src, 1
    return (a[0], a[1], b[0], b[1], packet.protocol), direction


def directional_flow(packet: PacketRecord) -> tuple[tuple[object, ...], int]:
    """Return a TFusion-compatible directed five-tuple.

    Endpoint A is the packet source and endpoint B is its destination. Because
    the reverse direction has a different key, every packet in one directional
    segment occupies the A-to-B channels of the unchanged 30 x 6 tensor.
    """

    return (
        packet.src_ip,
        int(packet.src_port),
        packet.dst_ip,
        int(packet.dst_port),
        packet.protocol,
    ), 0


def packet_flow(
    packet: PacketRecord, flow_orientation: str
) -> tuple[tuple[object, ...], int]:
    """Resolve a packet to the configured logical-flow key and channel."""

    if flow_orientation == "directional":
        return directional_flow(packet)
    if flow_orientation == "bidirectional":
        return canonical_flow(packet)
    raise ValueError(f"Unsupported flow orientation: {flow_orientation!r}")


def window_coordinates(
    timestamp: float,
    *,
    origin: float,
    window_seconds: float,
    window_alignment: str,
) -> tuple[int, float, float]:
    """Return window index, segment start, and offset within the window."""

    if window_alignment == "epoch":
        # Match TFusion's half-open epoch buckets [kW, (k+1)W).
        window_index = math.floor(timestamp / window_seconds)
        segment_start = window_index * window_seconds
    elif window_alignment == "capture":
        relative = max(0.0, timestamp - origin)
        window_index = math.floor(relative / window_seconds + 1e-12)
        segment_start = origin + window_index * window_seconds
    else:
        raise ValueError(f"Unsupported window alignment: {window_alignment!r}")
    return window_index, segment_start, max(0.0, timestamp - segment_start)


def build_capture_segments(
    spec: CaptureSpec,
    *,
    reader_path: Path,
    allowed_protocols: set[str],
    window_seconds: float,
    micro_bin_seconds: float,
    min_packets: int,
    flow_orientation: str = "bidirectional",
    window_alignment: str = "capture",
    microbin_feature_profile: str = "bidirectional_basic_v1",
) -> list[SegmentRecord]:
    started = time.perf_counter()
    packets = iter_packets(
        spec.path,
        reader_path=reader_path,
        allowed_protocols=allowed_protocols,
        start_frame=spec.start_frame,
        end_frame=spec.end_frame,
    )
    first = next(packets, None)
    if first is None:
        return []
    origin = first.timestamp
    num_bins = int(round(window_seconds / micro_bin_seconds))
    accumulators: "OrderedDict[tuple[object, ...], MicroBinAccumulator | DirectionalIATAccumulator]" = OrderedDict()
    attack_segments: dict[tuple[object, ...], bool] = {}
    segment_starts: dict[tuple[object, ...], float] = {}
    last_flow_timestamps: dict[tuple[object, ...], float] = {}
    labels = PacketAttackLabelStream(spec.packet_labels)

    def consume(packet: PacketRecord) -> None:
        flow_key, direction = packet_flow(packet, flow_orientation)
        previous_timestamp = last_flow_timestamps.get(flow_key)
        iat_seconds = (
            None
            if previous_timestamp is None
            else max(0.0, packet.timestamp - previous_timestamp)
        )
        last_flow_timestamps[flow_key] = packet.timestamp
        window_index, segment_start, within_window = window_coordinates(
            packet.timestamp,
            origin=origin,
            window_seconds=window_seconds,
            window_alignment=window_alignment,
        )
        bin_index = min(num_bins - 1, int(np.floor(within_window / micro_bin_seconds + 1e-12)))
        key = (window_index, *flow_key)
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = make_accumulator(num_bins, microbin_feature_profile)
            accumulators[key] = accumulator
        segment_starts.setdefault(key, segment_start)
        accumulator.add(
            bin_index,
            direction,
            packet.wire_length,
            iat_seconds=iat_seconds,
        )
        if labels.is_attack(packet.frame_number):
            attack_segments[key] = True

    try:
        consume(first)
        for packet in packets:
            consume(packet)
    finally:
        labels.close()

    capture = str(spec.path.resolve())
    capture_id = hashlib.sha1(capture.encode("utf-8")).hexdigest()[:16]
    records: list[SegmentRecord] = []
    for key, accumulator in accumulators.items():
        if accumulator.packet_count < min_packets:
            continue
        window_index, a_ip, a_port, b_ip, b_port, protocol = key
        flow_identity = f"{capture}|{a_ip}|{a_port}|{b_ip}|{b_port}|{protocol}"
        flow_id = hashlib.sha1(flow_identity.encode("utf-8")).hexdigest()
        segment_identity = f"{flow_identity}|{window_index}"
        records.append(
            SegmentRecord(
                features=accumulator.finalize(),
                segment_id=hashlib.sha1(segment_identity.encode("utf-8")).hexdigest(),
                flow_id=flow_id,
                capture=capture,
                capture_id=capture_id,
                split=spec.split,
                label_name="attack" if attack_segments.get(key, False) else spec.label,
                endpoint_a_ip=str(a_ip),
                endpoint_b_ip=str(b_ip),
                endpoint_a_port=int(a_port),
                endpoint_b_port=int(b_port),
                protocol=str(protocol),
                window_index=int(window_index),
                segment_start=float(segment_starts[key]),
                packet_count=accumulator.packet_count,
                byte_count=accumulator.byte_count,
                dataset_name=spec.dataset_name or spec.path.parent.name,
            )
        )
    records.sort(key=lambda row: (row.window_index, row.flow_id))
    elapsed = time.perf_counter() - started
    packets_read = sum(row.packet_count for row in records)
    print(
        f"parsed dataset={spec.dataset_name or spec.path.stem} eligible_packets={packets_read} "
        f"seconds={elapsed:.1f} packets_per_second={packets_read / max(elapsed, 1e-12):.1f}"
    )
    return records


def split_train_calibration(
    records: list[SegmentRecord], calibration_fraction: float
) -> list[SegmentRecord]:
    """Chronologically reserve the tail of one benign capture for calibration."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("data.calibration_fraction must be between zero and one")
    windows = sorted({row.window_index for row in records})
    if len(windows) < 2:
        raise ValueError("A train_calibration capture needs at least two active windows")
    calibration_windows = max(1, int(round(len(windows) * calibration_fraction)))
    cutoff = windows[-calibration_windows]
    return [
        replace(row, split="calibration" if row.window_index >= cutoff else "train")
        for row in records
    ]
