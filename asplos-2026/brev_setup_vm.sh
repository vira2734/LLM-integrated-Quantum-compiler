#!/usr/bin/env bash
set -euo pipefail

# NVIDIA Brev VM setup script (no env vars).
# This script mirrors the steps in BREV-SETUP.md:
# 1) Open-WBO-Inc setup (deps, submodule, build, verify)
# 2) satmapenv setup (python3.11 venv, requirements.txt)
# 3) Register satmapenv as a Jupyter kernel (--user)

REPO_URL="https://github.com/vira2734/LLM-integrated-Quantum-compiler.git"
REPO_DIR="LLM-integrated-Quantum-compiler"

if [[ -f "BREV-SETUP.md" ]] && [[ -f "requirements.txt" ]] && [[ -d "lib" ]]; then
  echo "[brev_setup_vm] repo root detected: $(pwd)"
else
  echo "[brev_setup_vm] not in repo root; ensuring repo is present in \$HOME"
  cd "$HOME"

  if [[ -d "${REPO_DIR}" ]]; then
    echo "[brev_setup_vm] repo already present: $HOME/${REPO_DIR}"
  else
    echo "[brev_setup_vm] cloning repo: ${REPO_URL}"
    git clone "${REPO_URL}"
  fi

  cd "$HOME/${REPO_DIR}"

  if [[ ! -f "BREV-SETUP.md" ]] || [[ ! -f "requirements.txt" ]] || [[ ! -d "lib" ]]; then
    echo "Error: expected repo root at $HOME/${REPO_DIR} but required files/dirs are missing." >&2
    exit 1
  fi
fi

echo "[brev_setup_vm] 1) Installing system dependencies"
sudo apt-get update
sudo apt-get install -y build-essential make libgmp-dev git

echo "[brev_setup_vm] 2) Initializing submodules"
git submodule update --init --recursive

echo "[brev_setup_vm] 3) Building Open-WBO-Inc (make r)"
(cd lib/Open-WBO-Inc && make r)

echo "[brev_setup_vm] 4) Verifying solver binary"
./lib/Open-WBO-Inc/open-wbo-inc_release --help >/dev/null

echo "[brev_setup_vm] 5) Installing Python 3.11 + venv tooling"
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

echo "[brev_setup_vm] 6) Creating satmapenv and installing requirements"
python3.11 -m venv satmapenv
# shellcheck disable=SC1090
source satmapenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[brev_setup_vm] 7) Registering satmapenv as a Jupyter kernel"
pip install ipykernel
python -m ipykernel install --user --name satmapenv --display-name "Python (satmapenv)"
deactivate

cat <<'EOF'
[brev_setup_vm] done

Restart Jupyter (or refresh the launcher), then select:
  Python (satmapenv)
EOF

