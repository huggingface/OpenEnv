#!/usr/bin/env python3
"""Verify complete evidence checksums and successful test execution."""

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree


def verify(root):
    root = root.resolve()
    recorded = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or relative in recorded:
            raise ValueError(f"Invalid artifact path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
        recorded[relative] = checksum
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    }
    if set(recorded) != actual:
        raise ValueError("Artifact inventory differs from SHA256SUMS")
    manifest = json.loads((root / "run-manifest.json").read_text())
    if not manifest["success"]:
        raise ValueError("Run did not complete successfully")
    suites = ElementTree.parse(root / "junit.xml").getroot().iter("testsuite")
    count = 0
    for suite in suites:
        count += int(suite.get("tests", "0"))
        if any(int(suite.get(key, "0")) for key in ("failures", "errors", "skipped")):
            raise ValueError("Required test suite contains failures, errors or skips")
    if not count:
        raise ValueError("Empty test suite cannot establish completion")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    print(f"Verified {verify(arguments.path)} tests and all artifact hashes")
