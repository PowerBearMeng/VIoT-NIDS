"""Offline packet labels; these never enter model inputs or training."""
from __future__ import annotations
import csv
from pathlib import Path

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
