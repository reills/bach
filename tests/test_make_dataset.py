from pathlib import Path

from scripts.make_dataset import _piece_id_for_path, _sha256_file, _source_path_for_path


def test_piece_id_for_path_uses_relative_path_to_avoid_stem_collisions():
    input_dir = Path("data/tobis_xml")
    first = input_dir / "folder-a" / "BWV_0131a" / "BWV_0131a.xml"
    second = input_dir / "folder-b" / "BWV_0131a" / "BWV_0131a.xml"

    assert _piece_id_for_path(first, input_dir) == "folder-a__BWV_0131a__BWV_0131a"
    assert _piece_id_for_path(second, input_dir) == "folder-b__BWV_0131a__BWV_0131a"
    assert _piece_id_for_path(first, input_dir) != _piece_id_for_path(second, input_dir)


def test_source_path_for_path_is_relative_to_input_dir():
    input_dir = Path("data/tobis_xml")
    path = input_dir / "folder-a" / "BWV_0131a" / "BWV_0131a.xml"

    assert _source_path_for_path(path, input_dir) == "folder-a/BWV_0131a/BWV_0131a.xml"


def test_sha256_file_hashes_file_bytes(tmp_path):
    path = tmp_path / "piece.xml"
    path.write_bytes(b"<score-partwise/>")

    assert _sha256_file(path) == "19360256545a24f2ff8741af4c9c73ff56cd0d69fd4d677b01a9661104365092"
