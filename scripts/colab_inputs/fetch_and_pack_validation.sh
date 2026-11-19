#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p staging
rsync -av --files-from=remote_files.txt pkandpal@discovery.usc.edu:/ staging/
python3 pack_bundle.py
