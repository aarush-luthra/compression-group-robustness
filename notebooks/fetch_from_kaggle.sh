#!/usr/bin/env bash
# After the Kaggle run finishes: download its outputs, copy the results
# (CSV + plot) into results/ and the checkpoints into checkpoints/.
# Then commit results/ to publish them on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
# Kaggle username: $KAGGLE_USERNAME, else ~/.kaggle/kaggle.json (legacy tokens).
# New KGAT_ tokens live in ~/.kaggle/access_token and don't include the username.
USER_NAME="${KAGGLE_USERNAME:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null || true)}"
[ -n "$USER_NAME" ] || { echo "Set KAGGLE_USERNAME=<your kaggle username>"; exit 1; }
KERNEL="$USER_NAME/waterbirds-phase-1-erm"
kaggle kernels status "$KERNEL"
rm -rf kaggle_output && mkdir -p kaggle_output
kaggle kernels output "$KERNEL" -p kaggle_output
mkdir -p results checkpoints
find kaggle_output -path "*results/*" -type f \( -name "*.csv" -o -name "*.png" \) -exec cp {} results/ \;
find kaggle_output -name "erm_seed*.pt" -exec cp {} checkpoints/ \;
ls -la results checkpoints
echo
echo "To publish results:  git add results && git commit -m 'Phase 1 ERM results' && git push"
