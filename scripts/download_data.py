"""
Download the SGCC electricity-theft dataset to data/sgcc_full.csv.

The dataset (42,372 customers, daily kWh for 2014-01-01..2016-10-31, FLAG=1 for
theft) is mirrored as a split zip in github.com/henryRDlab/ElectricityTheftDetection.
No credentials needed.

    python scripts/download_data.py [--output data/sgcc_full.csv]
"""

import argparse
import hashlib
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/henryRDlab/ElectricityTheftDetection/master/"
PARTS = ("data.z01", "data.z02", "data.zip")  # spanned-archive order
EXPECTED_SIZE = 175_194_613
EXPECTED_SHA256 = "99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7"


def fetch(name: str) -> bytes:
    print(f"Downloading {name}...", flush=True)
    with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
        return response.read()


def inflate_single_entry(archive: bytes) -> bytes:
    """Inflate the one deflated entry of a concatenated spanned zip (Python's zipfile can't read these)."""
    offset = archive.find(b"PK\x03\x04")
    if offset < 0:
        raise ValueError("No local file header found in archive")
    _, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", archive[offset:offset + 30])
    if method != 8:
        raise ValueError(f"Unsupported compression method {method}")
    start = offset + 30 + name_len + extra_len
    return zlib.decompressobj(-15).decompress(archive[start:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "data" / "sgcc_full.csv"))
    args = parser.parse_args()
    output = Path(args.output)

    if output.exists() and output.stat().st_size == EXPECTED_SIZE:
        print(f"{output} already present")
        return 0

    data = inflate_single_entry(b"".join(fetch(part) for part in PARTS))
    if len(data) != EXPECTED_SIZE:
        print(f"Unexpected size {len(data)} (expected {EXPECTED_SIZE})", file=sys.stderr)
        return 1
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"Checksum mismatch: {digest}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"Wrote {output} ({len(data) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
