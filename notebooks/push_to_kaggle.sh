#!/usr/bin/env bash
# Push phase1_kaggle.ipynb to Kaggle as a PRIVATE GPU notebook and start a run.
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
set -euo pipefail
cd "$(dirname "$0")"
USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
tmp=$(mktemp -d)
cp phase1_kaggle.ipynb "$tmp/"
sed "s/KAGGLE_USERNAME/$USER_NAME/" kernel-metadata.json > "$tmp/kernel-metadata.json"
kaggle kernels push -p "$tmp"
echo "Status:  kaggle kernels status $USER_NAME/waterbirds-phase1-erm"
echo "Outputs: kaggle kernels output $USER_NAME/waterbirds-phase1-erm -p ../kaggle_output"
