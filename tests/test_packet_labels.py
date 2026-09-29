from pathlib import Path

from data.flow_builder import PacketAttackLabelStream


def _read(path: Path, frames: list[int]) -> list[bool]:
    labels = PacketAttackLabelStream(path)
    try:
        return [labels.is_attack(frame) for frame in frames]
    finally:
        labels.close()


def test_gotham_packet_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("frame_number,binary_label\n1,0\n2,1\n3,0\n")
    assert _read(path, [1, 2, 3]) == [False, True, False]


def test_official_kitsune_r_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text('\"\",\"x\"\n\"1\",0\n\"2\",1\n\"3\",0\n')
    assert _read(path, [1, 2, 3]) == [False, True, False]


def test_headerless_kitsune_labels_allow_skipped_frames(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("0\n1\n0\n1\n")
    assert _read(path, [1, 4]) == [False, True]
