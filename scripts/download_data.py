"""下载 NASA C-MAPSS 数据集，并把文本文件解压到 data/raw/。

用法：python scripts/download_data.py

官方压缩包可能是嵌套的（外层 zip 里还有一层 zip），所以这里递归查找。
只按白名单提取文件，并且只用文件名本身作为目标路径，压缩包里的目录结构
不会影响写入位置（防止 zip slip 路径穿越）。
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

# NASA Prognostics Center of Excellence 数据库页面链接的下载地址
URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_FILES = frozenset(
    {f"{kind}_{subset}.txt" for kind in ("train", "test", "RUL") for subset in SUBSETS}
    | {"readme.txt"}
)


def extract_expected(archive: bytes, dest: Path) -> set[str]:
    """从（可能嵌套的）zip 字节中提取白名单文件，返回已提取的文件名。"""
    found: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            basename = name.replace("\\", "/").rsplit("/", 1)[-1]
            if basename.lower().endswith(".zip"):
                found |= extract_expected(zf.read(name), dest)
            elif basename in EXPECTED_FILES:
                (dest / basename).write_bytes(zf.read(name))
                found.add(basename)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 NASA C-MAPSS 数据集")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    print(f"下载 {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    print(f"已下载 {len(archive) / 1e6:.1f} MB")

    args.dest.mkdir(parents=True, exist_ok=True)
    found = extract_expected(archive, args.dest)
    missing = EXPECTED_FILES - found
    if missing:
        raise SystemExit(f"压缩包缺少文件：{sorted(missing)}")
    print(f"已提取 {len(found)} 个文件到 {args.dest}")


if __name__ == "__main__":
    main()
