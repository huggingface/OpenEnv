#!/usr/bin/env python3

"""
Download a standard hg38/GRCh38 reference (GENCODE) for RNA-seq benchmarks.

Why GENCODE?
  - widely used in publications
  - consistent gene models for human RNA-seq

This script downloads:
  - Genome FASTA (primary assembly) (for STAR)
  - Annotation GTF (for featureCounts and transcript→gene mapping)
  - Transcriptome FASTA (for Salmon)

It writes into an output directory and prints the resulting paths.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gencode-release", type=str, default="47")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = str(args.gencode_release).strip().lstrip("v")
    # GENCODE paths are stable; use the "current" release number.
    fasta_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"GRCh38.primary_assembly.genome.fa.gz"
    )
    gtf_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"gencode.v{rel}.annotation.gtf.gz"
    )
    tx_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"gencode.v{rel}.transcripts.fa.gz"
    )

    fasta_path = out_dir / "GRCh38.primary_assembly.genome.fa.gz"
    gtf_path = out_dir / f"gencode.v{rel}.annotation.gtf.gz"
    tx_path = out_dir / f"gencode.v{rel}.transcripts.fa.gz"

    if not fasta_path.exists():
        _run(["curl", "-L", "-o", str(fasta_path), fasta_url])
    if not gtf_path.exists():
        _run(["curl", "-L", "-o", str(gtf_path), gtf_url])
    if not tx_path.exists():
        _run(["curl", "-L", "-o", str(tx_path), tx_url])

    print("[OK] FASTA:", fasta_path)
    print("[OK]  GTF:", gtf_path)
    print("[OK]   TX:", tx_path)
    # Lightweight integrity signal (not an official checksum).
    print("[SHA256] FASTA:", _sha256(fasta_path))
    print("[SHA256]  GTF:", _sha256(gtf_path))
    print("[SHA256]   TX:", _sha256(tx_path))


if __name__ == "__main__":
    # Make curl retries a bit more robust if user sets it.
    os.environ.setdefault("CURL_RETRY", "5")
    main()

