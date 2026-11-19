#!/usr/bin/env bash
# Push phase1_kaggle.ipynb to Kaggle as a PRIVATE GPU notebook and start a run.
# Needs a Kaggle API token: ~/.kaggle/access_token (KGAT_...) or ~/.kaggle/kaggle.json.
set -euo pipefail
cd "$(dirname "$0")"
# Kaggle username: $KAGGLE_USERNAME, else ~/.kaggle/kaggle.json (legacy tokens).
# New KGAT_ tokens live in ~/.kaggle/access_token and don't include the username.
USER_NAME="${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null || true)}"
[ -n "$USER_NAME" ] || { echo "Set KAGGLE_USERNAME=<your kaggle username>"; exit 1; }
tmp=$(mktemp -d)
cp phase1_kaggle.ipynb "$tmp/"
sed "s/KAGGLE_USERNAME/$USER_NAME/" kernel-metadata.json > "$tmp/kernel-metadata.json"
kaggle kernels push -p "$tmp"
echo "Status:  kaggle kernels status $USER_NAME/waterbirds-phase-1-erm"
echo "Outputs: kaggle kernels output $USER_NAME/waterbirds-phase-1-erm -p ../kaggle_output"
