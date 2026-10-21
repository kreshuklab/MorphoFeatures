#!/usr/bin/env bash
# Optional overlay environment: reuses the existing scientific stack, adds MOFA2 locally.
set -euo pipefail
base_python="${1:-/g/kreshuk/frazer/miniforge3/envs/MorphoFeats_dev/bin/python}"
target_environment="${2:-/g/kreshuk/frazer/envs/morpho_shared_space}"
if [ -e "$target_environment" ]; then
    echo "Environment already exists: $target_environment (left unchanged)."
    exit 0
fi
"$base_python" -m venv --system-site-packages "$target_environment"
"$target_environment/bin/python" -m pip install 'mofapy2==0.7.3'
echo "Select $target_environment/bin/python as the notebook kernel interpreter."
echo "This environment inherits the base packages; each analysis saves pip freeze and versions."
