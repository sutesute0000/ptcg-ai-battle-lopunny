#!/bin/bash
# Pack main.py + deck.csv + cg/ into submission.tar.gz (all at top level).
set -euo pipefail
cd "$(dirname "$0")"
rm -f submission.tar.gz
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='.DS_Store' \
    -czf submission.tar.gz main.py deck.csv cg
ls -lh submission.tar.gz
tar -tzf submission.tar.gz | head -8
