#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Reuse the existing 150 image downloads. Read-only access to CARC.
mkdir -p ../csev_colab_validation/staging
rsync -av --files-from=remote_files.txt pkandpal@discovery.usc.edu:/ ../csev_colab_validation/staging/
python3 pack_full.py
