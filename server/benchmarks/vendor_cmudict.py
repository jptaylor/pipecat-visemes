"""Rebuild the vendored lexicon from a pinned upstream revision (dev-only)."""

import gzip
import hashlib
import json
import re
import struct
import urllib.request
from pathlib import Path

from lipsync.pronunciation import PHONES

REVISION = "74790861f652b15e4ac49015a90074ad62a27690"
BASE = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{REVISION}/"


def main():
    source = urllib.request.urlopen(BASE + "cmudict.dict", timeout=30).read()
    license_text = urllib.request.urlopen(BASE + "LICENSE", timeout=30).read()
    entries = {}
    for line in source.decode().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields or not re.fullmatch(r"[a-z]+(?:'[a-z]+)*", fields[0]):
            continue
        phones = [p.rstrip("012") for p in fields[1:]]
        if phones and all(p in PHONES for p in phones):
            entries.setdefault(fields[0], bytes(PHONES.index(p) + 1 for p in phones))
    header = 12 + 4 * len(entries)
    records, offsets = bytearray(), []
    for word, phones in sorted(entries.items()):
        offsets.append(header + len(records))
        records.extend(word.encode() + b"\0" + phones + b"\0")
    packed = (
        b"LPCMUD1\0"
        + struct.pack("<I", len(entries))
        + struct.pack(f"<{len(entries)}I", *offsets)
        + records
    )
    out = Path(__file__).resolve().parents[1] / "lipsync" / "data"
    out.mkdir(exist_ok=True)
    (out / "cmudict.bin.gz").write_bytes(gzip.compress(packed, mtime=0))
    (out / "CMUDICT-LICENSE").write_bytes(license_text)
    metadata = {
        "source": BASE + "cmudict.dict",
        "revision": REVISION,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "entries": len(entries),
        "packed_bytes": len(packed),
        "gzip_bytes": (out / "cmudict.bin.gz").stat().st_size,
    }
    (out / "cmudict.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
