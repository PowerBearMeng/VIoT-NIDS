#!/usr/bin/env python3
"""Generate deterministic payload-opaque PCAP fixtures for the smoke pipeline."""

from __future__ import annotations

import argparse
import ipaddress
import struct
from pathlib import Path


def _udp_frame(src: str, dst: str, src_port: int, dst_port: int, payload_size: int) -> bytes:
    source_mac = bytes.fromhex("020000000001")
    destination_mac = bytes.fromhex("020000000002")
    ethernet = destination_mac + source_mac + struct.pack("!H", 0x0800)
    total_length = 20 + 8 + payload_size
    ipv4 = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_length,
        0,
        0,
        64,
        17,
        0,
        ipaddress.IPv4Address(src).packed,
        ipaddress.IPv4Address(dst).packed,
    )
    udp = struct.pack("!HHHH", src_port, dst_port, 8 + payload_size, 0)
    # Opaque bytes exist only to produce realistic captured lengths. The NIDS
    # parser deliberately drops them before feature extraction.
    return ethernet + ipv4 + udp + bytes([0xA5]) * payload_size


def _write_pcap(path: Path, records: list[tuple[float, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for timestamp, frame in sorted(records):
            seconds = int(timestamp)
            micros = int(round((timestamp - seconds) * 1_000_000))
            handle.write(struct.pack("<IIII", seconds, micros, len(frame), len(frame)))
            handle.write(frame)


def _normal_records(start: float, windows: int, phase: int) -> list[tuple[float, bytes]]:
    records: list[tuple[float, bytes]] = []
    hub = "10.0.0.1"
    for window in range(windows):
        base = start + window * 3.0
        for device_index in range(2, 5):
            device = f"10.0.0.{device_index}"
            port = 5000 + device_index
            jitter = 0.02 * ((window + device_index + phase) % 4)
            records.append((base + 0.12 + jitter, _udp_frame(hub, device, 40000 + device_index, port, 30 + device_index)))
            records.append((base + 0.38 + jitter, _udp_frame(device, hub, port, 40000 + device_index, 24 + device_index)))
            records.append((base + 1.15 + jitter, _udp_frame(hub, device, 40000 + device_index, port, 34 + (window % 3))))
            if (window + device_index + phase) % 3 == 0:
                records.append((base + 2.22, _udp_frame(device, hub, port, 40000 + device_index, 20)))
    return records


def _attack_records(start: float, windows: int) -> list[tuple[float, bytes]]:
    records = _normal_records(start, windows, phase=5)
    attacker = "10.0.0.99"
    for window in range(windows):
        base = start + window * 3.0
        for victim_index in range(2, 6):
            victim = f"10.0.0.{victim_index}"
            src_port = 45000 + victim_index
            dst_port = 8000 + victim_index
            for packet_index in range(14):
                timestamp = base + 0.05 + packet_index * 0.055 + victim_index * 0.002
                records.append((timestamp, _udp_frame(attacker, victim, src_port, dst_port, 600 + 10 * victim_index)))
            records.append((base + 1.1, _udp_frame(victim, attacker, dst_port, src_port, 16)))
    return records


def generate(root: Path) -> list[Path]:
    specifications = [
        ("train/normal/train_normal.pcap", _normal_records(1_700_000_000.0, 14, 0)),
        ("calibration/normal/calibration_normal.pcap", _normal_records(1_700_100_000.0, 8, 2)),
        ("test/normal/test_normal.pcap", _normal_records(1_700_200_000.0, 8, 4)),
        ("test/attack/test_attack.pcap", _attack_records(1_700_300_000.0, 8)),
    ]
    outputs: list[Path] = []
    for relative, records in specifications:
        output = root / relative
        _write_pcap(output, records)
        outputs.append(output)
        print(f"generated {output} packets={len(records)}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke")
    args = parser.parse_args()
    generate(Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
