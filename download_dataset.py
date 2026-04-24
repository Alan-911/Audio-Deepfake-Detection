"""
CompSpoofV2 Dataset Download
=============================
Downloads the CompSpoofV2 dataset for the ESDD2 challenge.

Dataset page : https://xuepingzhang.github.io/CompSpoof-V2-Dataset/
Challenge    : https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-2
Size         : ~283 hours, ~50 GB — ensure sufficient disk space before running.

Usage:
    python download_dataset.py
    python download_dataset.py --out_dir /path/to/data/CompSpoofV2

Manual download (if automatic fails):
    1. Visit https://xuepingzhang.github.io/CompSpoof-V2-Dataset/
    2. Follow the access request instructions on the dataset page.
    3. Place files under ./data/CompSpoofV2/ matching the structure below.

Expected structure after download:
    data/CompSpoofV2/
    ├── development/
    │   ├── train.csv
    │   ├── val.csv
    │   └── audio/
    └── eval/
        └── metadata/
            └── eval.csv
"""

import os
import argparse


def download_esdd2(out_dir: str = "./data/CompSpoofV2"):
    """
    Attempt to download CompSpoofV2 from HuggingFace Hub.
    Falls back with clear manual instructions if the dataset is not yet public.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("[*] CompSpoofV2 Dataset Download")
    print(f"[*] Target directory : {out_dir}")
    print("[*] Dataset size     : ~283 hours (~50 GB). Ensure sufficient disk space.")
    print()

    # Primary: attempt HuggingFace Hub download
    # The dataset may require access approval — check the dataset page first.
    HF_REPO_ID = "XuepingZhang/CompSpoofV2"   # update if repo ID changes

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        print(f"[*] Attempting HuggingFace Hub download: {HF_REPO_ID}")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=out_dir,
        )
        print(f"[*] Download complete. Data ready in: {out_dir}")

    except Exception as e:
        print(f"[!] HuggingFace download failed: {e}")
        print()
        print("=" * 60)
        print("  MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("  1. Visit the dataset page:")
        print("     https://xuepingzhang.github.io/CompSpoof-V2-Dataset/")
        print()
        print("  2. Request access or follow the download instructions.")
        print()
        print("  3. Extract files to:")
        print(f"     {os.path.abspath(out_dir)}")
        print()
        print("  4. Expected structure:")
        print("     data/CompSpoofV2/")
        print("     ├── development/")
        print("     │   ├── train.csv")
        print("     │   ├── val.csv")
        print("     │   └── audio/")
        print("     └── eval/")
        print("         └── metadata/")
        print("             └── eval.csv")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CompSpoofV2 dataset")
    parser.add_argument("--out_dir", type=str, default="./data/CompSpoofV2",
                        help="Directory to download dataset into")
    args = parser.parse_args()
    download_esdd2(args.out_dir)
