# Compression vs. group robustness (Waterbirds, ResNet-18)

Phase 1: baseline pipeline (data, eval harness, ERM ResNet-18).

## Kaggle
Open `notebooks/phase1_kaggle.ipynb`, set GPU, and either attach Waterbirds as an
input dataset or turn Internet on (the notebook downloads it). It verifies the
data, trains 3 seeds and summarizes test results.

## Data
```
# download (~492 MB) from the Hugging Face mirror grodino/waterbirds (a 1:1
# conversion of the official release) + verify split/group counts and files
python -m src.prepare_data --data_root data --download
# -> dataset root is data/waterbirds
# (--source stanford uses the original tarball URL, which is often unreachable)
# or verify an existing copy
python -m src.prepare_data --data_root /path/to/waterbird_complete95_forest2water2
```
Expected train group counts: 3498 / 184 / 56 / 1057 (val 467/466/133/133,
test 2255/2255/642/642).

## Run (local)
```
pip install -r requirements.txt
python -m src.train_erm --data_root /path/to/waterbirds_v1.0 --seed 0
python -m src.train_erm --data_root /path/to/waterbirds_v1.0 --seed 1
python -m src.train_erm --data_root /path/to/waterbirds_v1.0 --seed 2
pytest tests
```
Results append to `results/results.csv`
(`model,seed,split,overall,g0..g3,wga,size_mb,latency_ms`).

Groups: `g = 2*y + place` -> g0 landbird/land, g1 landbird/water,
g2 waterbird/land, g3 waterbird/water. Rare: g1 and g2.

Rules: select checkpoints by val WGA; test only for final eval; latency always
measured on CPU, 1 thread, batch 1 (INT8 is CPU-only).
