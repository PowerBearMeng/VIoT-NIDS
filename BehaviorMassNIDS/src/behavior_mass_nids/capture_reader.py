"""Dependency-free PCAP/PCAPNG reader and packet-header parser.

The paper uses the first M raw header bytes of TCP/UDP packets. This module
keeps link, IP, and transport headers while dropping the encrypted payload.
ARP, ICMP, non-initial fragments, malformed packets, and protocols without
ports are ignored.
"""

from __future__ import annotations

import ipaddress
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228
LINKTYPE_IPV6 = 229
LINKTYPE_LINUX_SLL2 = 276


@dataclass(frozen=True)
class CapturedPacket:
    timestamp: float
    data: bytes
    linktype: int


@dataclass(frozen=True)
class ParsedPacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    header: bytes


class CaptureFormatError(ValueError):
    """Raised when a capture file has an unsupported or corrupt format."""


def iter_capture(path: str | Path) -> Iterator[CapturedPacket]:
    """Yield packets from classic PCAP or PCAPNG captures."""
    capture_path = Path(path)
    with capture_path.open("rb") as handle:
        magic = handle.read(4)
        handle.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            yield from _iter_pcapng(handle)
        elif magic in {
            b"\xd4\xc3\xb2\xa1",
            b"\xa1\xb2\xc3\xd4",
            b"\x4d\x3c\xb2\xa1",
            b"\xa1\xb2\x3c\x4d",
        }:
            yield from _iter_pcap(handle)
        else:
            raise CaptureFormatError(f"Unsupported capture magic in {capture_path}")


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise CaptureFormatError("Unexpected end of capture file")
    return value


def _iter_pcap(handle: BinaryIO) -> Iterator[CapturedPacket]:
    header = _read_exact(handle, 24)
    magic = header[:4]
    little = magic in {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}
    nano = magic in {b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
    endian = "<" if little else ">"
    _, _, _, _, _, linktype = struct.unpack(endian + "HHiiii", header[4:])
    divisor = 1_000_000_000.0 if nano else 1_000_000.0

    while True:
        record_header = handle.read(16)
        if not record_header:
            break
        if len(record_header) != 16:
            raise CaptureFormatError("Truncated PCAP record header")
        ts_sec, ts_fraction, captured_len, _ = struct.unpack(
            endian + "IIII", record_header
        )
        packet = _read_exact(handle, captured_len)
        yield CapturedPacket(ts_sec + ts_fraction / divisor, packet, linktype)


def _iter_pcapng(handle: BinaryIO) -> Iterator[CapturedPacket]:
    endian = "<"
    interfaces: list[tuple[int, float]] = []
    section_seen = False

    while True:
        block_prefix = handle.read(8)
        if not block_prefix:
            break
        if len(block_prefix) != 8:
            raise CaptureFormatError("Truncated PCAPNG block header")

        raw_type = block_prefix[:4]
        if raw_type == b"\x0a\x0d\x0d\x0a":
            byte_order = _read_exact(handle, 4)
            if byte_order == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif byte_order == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                raise CaptureFormatError("Invalid PCAPNG byte-order magic")
            total_length = struct.unpack(endian + "I", block_prefix[4:])[0]
            if total_length < 28:
                raise CaptureFormatError("Invalid PCAPNG section length")
            # Eight block-prefix bytes and four byte-order bytes were consumed;
            # the remaining bytes include the trailing total-length field.
            body_rest = _read_exact(handle, total_length - 12)
            closing_length = struct.unpack(endian + "I", body_rest[-4:])[0]
            if closing_length != total_length:
                raise CaptureFormatError("Mismatched PCAPNG section length")
            interfaces = []
            section_seen = True
            continue

        if not section_seen:
            raise CaptureFormatError("PCAPNG data appears before section header")
        block_type, total_length = struct.unpack(endian + "II", block_prefix)
        if total_length < 12 or total_length % 4:
            raise CaptureFormatError("Invalid PCAPNG block length")
        rest = _read_exact(handle, total_length - 8)
        if struct.unpack(endian + "I", rest[-4:])[0] != total_length:
            raise CaptureFormatError("Mismatched PCAPNG block length")
        body = rest[:-4]

        if block_type == 1:  # Interface Description Block
            if len(body) < 8:
                raise CaptureFormatError("Truncated interface block")
            linktype = struct.unpack(endian + "H", body[:2])[0]
            ts_resolution = _pcapng_timestamp_resolution(body[8:], endian)
            interfaces.append((linktype, ts_resolution))
        elif block_type == 6:  # Enhanced Packet Block
            if len(body) < 20:
                raise CaptureFormatError("Truncated enhanced packet block")
            interface_id, ts_high, ts_low, captured_len, _ = struct.unpack(
                endian + "IIIII", body[:20]
            )
            if interface_id >= len(interfaces):
                raise CaptureFormatError("Packet references an unknown interface")
            padded_len = (captured_len + 3) & ~3
            if len(body) < 20 + padded_len:
                raise CaptureFormatError("Truncated enhanced packet payload")
            linktype, resolution = interfaces[interface_id]
            timestamp = ((ts_high << 32) | ts_low) * resolution
            yield CapturedPacket(
                timestamp, bytes(body[20 : 20 + captured_len]), linktype
            )
        elif block_type == 3:  # Simple Packet Block (no timestamp)
            if not interfaces or len(body) < 4:
                continue
            original_len = struct.unpack(endian + "I", body[:4])[0]
            captured_len = min(original_len, len(body) - 4)
            yield CapturedPacket(0.0, bytes(body[4 : 4 + captured_len]), interfaces[0][0])


def _pcapng_timestamp_resolution(options: bytes, endian: str) -> float:
    offset = 0
    while offset + 4 <= len(options):
        code, length = struct.unpack(endian + "HH", options[offset : offset + 4])
        offset += 4
        value = options[offset : offset + length]
        offset += (length + 3) & ~3
        if code == 0:
            break
        if code == 9 and value:
            raw = value[0]
            return 2.0 ** -(raw & 0x7F) if raw & 0x80 else 10.0 ** -raw
    return 1e-6


def parse_packet(
    packet: CapturedPacket,
    *,
    include_link_header: bool = True,
    allowed_protocols: set[str] | None = None,
) -> ParsedPacket | None:
    """Extract a unidirectional five-tuple and payload-free raw header."""
    allowed = allowed_protocols or {"tcp", "udp"}
    decoded = _network_offset(packet.data, packet.linktype)
    if decoded is None:
        return None
    network_offset, ethertype = decoded
    if ethertype == 0x0800:
        parsed = _parse_ipv4(packet.data, network_offset)
    elif ethertype == 0x86DD:
        parsed = _parse_ipv6(packet.data, network_offset)
    else:
        return None
    if parsed is None:
        return None

    src_ip, dst_ip, src_port, dst_port, protocol, header_end = parsed
    if protocol not in allowed or header_end > len(packet.data):
        return None
    header_start = 0 if include_link_header else network_offset
    return ParsedPacket(
        timestamp=packet.timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        header=packet.data[header_start:header_end],
    )


def _network_offset(data: bytes, linktype: int) -> tuple[int, int] | None:
    if linktype == LINKTYPE_ETHERNET:
        if len(data) < 14:
            return None
        offset = 14
        ethertype = struct.unpack("!H", data[12:14])[0]
        while ethertype in {0x8100, 0x88A8, 0x9100}:
            if len(data) < offset + 4:
                return None
            ethertype = struct.unpack("!H", data[offset + 2 : offset + 4])[0]
            offset += 4
        return offset, ethertype
    if linktype == LINKTYPE_LINUX_SLL:
        if len(data) < 16:
            return None
        return 16, struct.unpack("!H", data[14:16])[0]
    if linktype == LINKTYPE_LINUX_SLL2:
        if len(data) < 20:
            return None
        return 20, struct.unpack("!H", data[:2])[0]
    if linktype == LINKTYPE_RAW:
        if not data:
            return None
        version = data[0] >> 4
        return (0, 0x0800 if version == 4 else 0x86DD) if version in {4, 6} else None
    if linktype == LINKTYPE_IPV4:
        return 0, 0x0800
    if linktype == LINKTYPE_IPV6:
        return 0, 0x86DD
    if linktype == LINKTYPE_NULL:
        if len(data) < 4:
            return None
        family_le = struct.unpack("<I", data[:4])[0]
        family_be = struct.unpack(">I", data[:4])[0]
        if 2 in {family_le, family_be}:
            return 4, 0x0800
        if {family_le, family_be} & {10, 24, 28, 30}:
            return 4, 0x86DD
    return None


def _parse_ipv4(data: bytes, offset: int) -> tuple[str, str, int, int, str, int] | None:
    if len(data) < offset + 20 or data[offset] >> 4 != 4:
        return None
    ihl = (data[offset] & 0x0F) * 4
    if ihl < 20 or len(data) < offset + ihl:
        return None
    fragment = struct.unpack("!H", data[offset + 6 : offset + 8])[0]
    if fragment & 0x1FFF:  # no transport header in non-initial fragment
        return None
    protocol_number = data[offset + 9]
    if protocol_number == 1:  # ICMP
        return None
    src = str(ipaddress.IPv4Address(data[offset + 12 : offset + 16]))
    dst = str(ipaddress.IPv4Address(data[offset + 16 : offset + 20]))
    return _parse_transport(data, offset + ihl, protocol_number, src, dst)


def _parse_ipv6(data: bytes, offset: int) -> tuple[str, str, int, int, str, int] | None:
    if len(data) < offset + 40 or data[offset] >> 4 != 6:
        return None
    next_header = data[offset + 6]
    cursor = offset + 40
    src = str(ipaddress.IPv6Address(data[offset + 8 : offset + 24]))
    dst = str(ipaddress.IPv6Address(data[offset + 24 : offset + 40]))

    while next_header in {0, 43, 44, 51, 60}:
        if next_header == 44:
            if len(data) < cursor + 8:
                return None
            fragment_offset = struct.unpack("!H", data[cursor + 2 : cursor + 4])[0] >> 3
            if fragment_offset:
                return None
            next_header, cursor = data[cursor], cursor + 8
        elif next_header == 51:
            if len(data) < cursor + 2:
                return None
            length = (data[cursor + 1] + 2) * 4
            next_header = data[cursor]
            cursor += length
        else:
            if len(data) < cursor + 2:
                return None
            length = (data[cursor + 1] + 1) * 8
            next_header = data[cursor]
            cursor += length
        if cursor > len(data):
            return None
    if next_header == 58:  # ICMPv6
        return None
    return _parse_transport(data, cursor, next_header, src, dst)


def _parse_transport(
    data: bytes, offset: int, protocol_number: int, src: str, dst: str
) -> tuple[str, str, int, int, str, int] | None:
    if protocol_number == 6:
        if len(data) < offset + 20:
            return None
        src_port, dst_port = struct.unpack("!HH", data[offset : offset + 4])
        tcp_header_len = (data[offset + 12] >> 4) * 4
        if tcp_header_len < 20 or len(data) < offset + tcp_header_len:
            return None
        return src, dst, src_port, dst_port, "tcp", offset + tcp_header_len
    if protocol_number == 17:
        if len(data) < offset + 8:
            return None
        src_port, dst_port = struct.unpack("!HH", data[offset : offset + 4])
        return src, dst, src_port, dst_port, "udp", offset + 8
    return None


def iter_parsed_packets(
    path: str | Path,
    *,
    include_link_header: bool = True,
    allowed_protocols: set[str] | None = None,
) -> Iterator[ParsedPacket]:
    for captured in iter_capture(path):
        parsed = parse_packet(
            captured,
            include_link_header=include_link_header,
            allowed_protocols=allowed_protocols,
        )
        if parsed is not None:
            yield parsed
