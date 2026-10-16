"""Check or rebuild the SHA-256 manifest of the versioned repository files."""

import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Rebuild from files in the Git index")
    args = parser.parse_args()
    if args.update:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        lines = []
        for name in sorted(name for name in names if name and name != MANIFEST.name):
            lines.append(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "  " + name)
        MANIFEST.write_text("\n".join(lines) + "\n")
        print(f"Wrote {len(lines)} file hashes.")
        return
    checked = 0
    for line in MANIFEST.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid manifest entry: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
        checked += 1
    print(f"Verified {checked} file hashes.")


if __name__ == "__main__":
    main()
