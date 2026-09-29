"""Read TCP/UDP packet metadata using the existing BPF-DAG parser.

Only timestamps, endpoint identifiers, transport protocol, and captured frame
length are exposed. Payload bytes and application-layer protocol semantics never
leave this adapter.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterator


@dataclass(frozen=True)
class PacketRecord:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    wire_length: int
    frame_number: int = 0


@lru_cache(maxsize=4)
def _load_reader(reader_path: str) -> ModuleType:
    path = Path(reader_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline PCAP reader not found at {path}. Set data.baseline_reader_path."
        )
    module_name = "_mymodel_bpf_dag_pcap_" + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PCAP reader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def iter_packets(
    pcap_path: str | Path,
    *,
    reader_path: str | Path,
    allowed_protocols: set[str] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Iterator[PacketRecord]:
    """Yield payload-independent metadata for supported TCP/UDP packets."""
    reader = _load_reader(str(Path(reader_path).expanduser().resolve()))
    protocols = allowed_protocols or {"tcp", "udp"}
    for frame_number, captured in enumerate(reader.iter_capture(Path(pcap_path)), start=1):
        if start_frame is not None and frame_number < start_frame:
            continue
        if end_frame is not None and frame_number > end_frame:
            break
        # Match TFusion's tshark ``ip.len`` semantics.  The capture used by
        # Kitsune has snaplen~=200, so len(captured.data) is only caplen and
        # severely truncates video packet sizes.  IPv4 total_length remains in
        # the captured IP header and is therefore available without payload.
        wire_length = _tfusion_ipv4_length(captured, reader)
        if wire_length is None:
            continue
        parsed = reader.parse_packet(
            captured,
            include_link_header=False,
            allowed_protocols=protocols,
        )
        if parsed is None:
            continue
        yield PacketRecord(
            timestamp=float(parsed.timestamp),
            src_ip=parsed.src_ip,
            dst_ip=parsed.dst_ip,
            src_port=int(parsed.src_port),
            dst_port=int(parsed.dst_port),
            protocol=parsed.protocol,
            wire_length=wire_length,
            frame_number=frame_number,
        )


def _tfusion_ipv4_length(captured: object, reader: ModuleType) -> int | None:
    """Return clamp(ip.len, 1, 1500), exactly as TFusion preprocessing does."""
    data = captured.data
    decoded = reader._network_offset(data, captured.linktype)
    if decoded is None:
        return None
    network_offset, ethertype = decoded
    if ethertype != 0x0800 or len(data) < network_offset + 4:
        return None
    total_length = struct.unpack("!H", data[network_offset + 2 : network_offset + 4])[0]
    if total_length < 20:
        return None
    return max(1, min(1500, int(total_length)))
