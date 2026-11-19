"""Download (optional) and verify the Waterbirds dataset.

Usage:
    # verify an existing copy (e.g. a Kaggle input dataset)
    python -m src.prepare_data --data_root /kaggle/input/<dataset>/waterbird_complete95_forest2water2

    # download the official release (~450 MB) and verify it
    python -m src.prepare_data --data_root data/waterbirds --download

Verification checks the official split sizes and per-group counts from
Sagawa et al. (2020), that the required metadata columns exist and that every
image file is present. Exits non-zero on any mismatch.
"""
import argparse
import os
import sys
import tarfile
import urllib.request

import pandas as pd

URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
ARCHIVE_DIR = "waterbird_complete95_forest2water2"
REQUIRED_COLS = {"img_filename", "y", "split", "place"}

# Official counts. group = 2*y + place
#   g0 landbird/land, g1 landbird/water, g2 waterbird/land, g3 waterbird/water
EXPECTED = {
    0: {"name": "train", "groups": [3498, 184, 56, 1057]},
    1: {"name": "val", "groups": [467, 466, 133, 133]},
    2: {"name": "test", "groups": [2255, 2255, 642, 642]},
}


def find_root(path):
    """Accept either the folder containing metadata.csv or its parent."""
    if os.path.isfile(os.path.join(path, "metadata.csv")):
        return path
    for dirpath, _, files in os.walk(path):
        if "metadata.csv" in files:
            return dirpath
    return None


def download(dest):
    os.makedirs(dest, exist_ok=True)
    tgz = os.path.join(dest, ARCHIVE_DIR + ".tar.gz")
    if not os.path.exists(tgz):
        print(f"Downloading {URL} -> {tgz}")
        urllib.request.urlretrieve(URL, tgz)
    print("Extracting...")
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(dest, filter="data")  # safe extraction (Python >= 3.11.4)
        except TypeError:
            tf.extractall(dest)  # older Python on some Kaggle images
    return os.path.join(dest, ARCHIVE_DIR)


HF_REPO = "grodino/waterbirds"  # 1:1 conversion of the official tarball (see its scripts/waterbirds.py)
HF_SPLITS = {"train": 0, "validation": 1, "test": 2}


def download_hf(dest):
    """Download the Hugging Face parquet mirror and unpack it into the official
    layout: <dest>/waterbirds/{train,val,test}/<n>.jpg + metadata.csv.
    Image bytes are written as-is (no re-encoding)."""
    import pyarrow.parquet as pq

    root = os.path.join(dest, "waterbirds")
    if os.path.isfile(os.path.join(root, "metadata.csv")):
        print("Already unpacked:", root)
        return root
    os.makedirs(root, exist_ok=True)
    rows = []
    for hf_split, split_id in HF_SPLITS.items():
        fname = f"{hf_split}-00000-of-00001.parquet"
        local = os.path.join(dest, fname)
        if not os.path.exists(local):
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/{fname}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, local + ".part")
            os.replace(local + ".part", local)
        table = pq.read_table(local, columns=["image", "label", "place"]).to_pydict()
        sub = ["train", "val", "test"][split_id]
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        for i, (img, y, place) in enumerate(zip(table["image"], table["label"], table["place"])):
            b = img["bytes"]
            ext = ".png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            rel = f"{sub}/{i:05d}{ext}"
            with open(os.path.join(root, rel), "wb") as f:
                f.write(b)
            rows.append({"img_filename": rel, "y": int(y), "split": split_id, "place": int(place)})
        print(f"  {sub}: {len(table['label'])} images")
    pd.DataFrame(rows).to_csv(os.path.join(root, "metadata.csv"), index=False)
    return root


def verify(root, check_files=True):
    """Return a list of error strings (empty = OK)."""
    errors = []
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    missing = REQUIRED_COLS - set(meta.columns)
    if missing:
        return [f"metadata.csv missing columns: {sorted(missing)}"]

    meta["group"] = 2 * meta["y"] + meta["place"]
    print(f"{'split':<6} {'total':>6}  g0(LL) g1(LW) g2(WL) g3(WW)")
    for split, exp in EXPECTED.items():
        sub = meta[meta["split"] == split]
        got = [int((sub["group"] == g).sum()) for g in range(4)]
        flag = "" if got == exp["groups"] else "  <-- MISMATCH"
        print(f"{exp['name']:<6} {len(sub):>6}  " + " ".join(f"{c:>6}" for c in got) + flag)
        if got != exp["groups"]:
            errors.append(f"{exp['name']}: group counts {got} != expected {exp['groups']}")

    if check_files:
        absent = [f for f in meta["img_filename"] if not os.path.isfile(os.path.join(root, f))]
        if absent:
            errors.append(f"{len(absent)} image files missing, e.g. {absent[:3]}")
        else:
            print(f"All {len(meta)} image files present.")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--source", choices=["stanford", "hf"], default="hf",
                    help="hf = Hugging Face mirror (default; the Stanford host is often down)")
    ap.add_argument("--skip_file_check", action="store_true")
    a = ap.parse_args()

    if a.download:
        root = download_hf(a.data_root) if a.source == "hf" else download(a.data_root)
    else:
        root = a.data_root
    root = find_root(root)
    if root is None:
        sys.exit(f"No metadata.csv found under {a.data_root}")
    print("Dataset root:", root)

    errors = verify(root, check_files=not a.skip_file_check)
    if errors:
        print("\nFAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("\nOK: Waterbirds matches the official splits. Use --data_root", root)


if __name__ == "__main__":
    main()
