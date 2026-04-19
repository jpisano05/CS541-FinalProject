##make sure it runs in bash
#!/bin/bash
set -euo pipefail

##get the right python
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python not found"
    exit 1
fi

##open environment
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

##run
$PYTHON train.py