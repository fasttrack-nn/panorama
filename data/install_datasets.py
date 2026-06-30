#!/usr/bin/env python3
"""
Download all .fvec dataset files from HuggingFace (PanoramaVLDB/data)
into a local directory.

Split files (e.g. openai_base.fvec.part00..03) are automatically
reassembled into the original .fvec after download.

Usage:
    python3 install_datasets.py /path/to/destination
    python3 install_datasets.py                        # defaults to ./datasets
"""

import argparse
import glob
import os
import sys

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "PanoramaVLDB/data"
REPO_TYPE = "dataset"
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB


def reassemble_parts(dest: str) -> None:
    """Find *.fvec.part00 files and concatenate all parts into the original .fvec."""
    part0_files = sorted(glob.glob(os.path.join(dest, "*.fvec.part00")))
    for part0 in part0_files:
        base = part0.rsplit(".part00", 1)[0]
        if os.path.exists(base):
            print(f"  {os.path.basename(base)} already exists, skipping reassembly")
            continue

        parts = sorted(glob.glob(base + ".part*"))
        total = sum(os.path.getsize(p) for p in parts)
        print(f"  Reassembling {len(parts)} parts into {os.path.basename(base)} ({total / (1024**3):.2f} GB) ...")

        with open(base, "wb") as out:
            for part_path in parts:
                with open(part_path, "rb") as inp:
                    while True:
                        buf = inp.read(CHUNK_SIZE)
                        if not buf:
                            break
                        out.write(buf)

        if os.path.getsize(base) != total:
            print(f"  ERROR: size mismatch after reassembly!", file=sys.stderr)
            sys.exit(1)

        for part_path in parts:
            os.remove(part_path)
        print(f"  Done: {os.path.basename(base)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Panorama benchmark datasets from Hugging Face Hub"
    )
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_dest = os.path.join(repo_root, "datasets")

    parser.add_argument(
        "destination",
        nargs="?",
        default=default_dest,
        help=f"Local directory to save files into (default: {default_dest})",
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.destination)
    os.makedirs(dest, exist_ok=True)
    print(f"Destination: {dest}")

    api = HfApi()
    repo_files = sorted(
        f
        for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        if f.endswith(".fvec") or ".fvec.part" in f
    )

    if not repo_files:
        print(f"No dataset files found in {REPO_ID}.")
        sys.exit(0)

    print(f"Found {len(repo_files)} files in {REPO_ID}:\n")

    for i, filename in enumerate(repo_files, 1):
        local_path = os.path.join(dest, filename)

        if os.path.exists(local_path):
            print(f"[{i}/{len(repo_files)}] {filename} already exists, skipping")
            continue

        print(f"[{i}/{len(repo_files)}] Downloading {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=dest,
        )
        print(f"  Saved to {local_path}")

    print("\nReassembling split files ...")
    reassemble_parts(dest)

    print(f"\nAll files downloaded to {dest}")


if __name__ == "__main__":
    main()
