import os

import pandas as pd

from src.prepare_data import EXPECTED, find_root, verify


def _fake_dataset(tmp_path, corrupt=False, drop_file=False):
    rows = []
    for split, exp in EXPECTED.items():
        for g, n in enumerate(exp["groups"]):
            if corrupt and split == 0 and g == 2:
                n -= 1
            for i in range(n):
                rows.append({"img_filename": f"s{split}/g{g}_{i}.jpg",
                             "y": g // 2, "place": g % 2, "split": split})
    root = tmp_path / "waterbirds"
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    for r in rows[: 5]:
        p = root / r["img_filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return root


def test_verify_ok_counts(tmp_path):
    root = _fake_dataset(tmp_path)
    assert verify(str(root), check_files=False) == []


def test_verify_detects_count_mismatch(tmp_path):
    root = _fake_dataset(tmp_path, corrupt=True)
    errs = verify(str(root), check_files=False)
    assert len(errs) == 1 and "train" in errs[0]


def test_verify_detects_missing_files(tmp_path):
    root = _fake_dataset(tmp_path)
    errs = verify(str(root), check_files=True)
    assert any("missing" in e for e in errs)


def test_find_root_in_parent(tmp_path):
    root = _fake_dataset(tmp_path)
    assert os.path.samefile(find_root(str(tmp_path)), root)
