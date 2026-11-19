#!/usr/bin/env bash
# After the Kaggle run finishes: download its outputs, copy the results
# (CSV + plot) into results/ and the checkpoints into checkpoints/.
# Then commit results/ to publish them on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
KERNEL="$USER_NAME/waterbirds-phase1-erm"
kaggle kernels status "$KERNEL"
rm -rf kaggle_output && mkdir -p kaggle_output
kaggle kernels output "$KERNEL" -p kaggle_output
mkdir -p results checkpoints
find kaggle_output -path "*results/*" -type f \( -name "*.csv" -o -name "*.png" \) -exec cp {} results/ \;
find kaggle_output -name "erm_seed*.pt" -exec cp {} checkpoints/ \;
ls -la results checkpoints
echo
echo "To publish results:  git add results && git commit -m 'Phase 1 ERM results' && git push"
