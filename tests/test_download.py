"""下载脚本的解压逻辑测试（不联网）。"""

import importlib.util
import io
import zipfile
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
_spec = importlib.util.spec_from_file_location("download_data", _SCRIPT)
download_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_data)


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_extracts_from_nested_zip(tmp_path):
    inner = _zip({"train_FD001.txt": b"data"})
    outer = _zip({"folder/CMAPSSData.zip": inner})

    found = download_data.extract_expected(outer, tmp_path)

    assert found == {"train_FD001.txt"}
    assert (tmp_path / "train_FD001.txt").read_bytes() == b"data"


def test_ignores_files_outside_whitelist(tmp_path):
    archive = _zip({"evil.py": b"x", "Damage Propagation Modeling.pdf": b"x"})

    assert download_data.extract_expected(archive, tmp_path) == set()
    assert list(tmp_path.iterdir()) == []


def test_path_traversal_stays_inside_dest(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive = _zip({"../../train_FD001.txt": b"data"})

    download_data.extract_expected(archive, dest)

    assert (dest / "train_FD001.txt").exists()
    assert not (tmp_path / "train_FD001.txt").exists()
