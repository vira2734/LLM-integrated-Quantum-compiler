# Setup on NVIDIA Brev

SAT solver and Python environment setup for an NVIDIA Brev launchable instance.

## 1. Open-WBO-Inc (SAT solver) setup

### 1.1 Get the repo

If the launchable didn't clone your repo, clone it (from the Brev terminal / SSH):

```bash
# If private repo (replace with your repo URL and token)
git clone https://github.com/vira2734/LLM-integrated-Quantum-compiler.git
cd LLM-integrated-Quantum-compiler
```

If the repo is already there:

```bash
cd /path/to/LLM-integrated-Quantum-compiler   # or wherever Brev put it, e.g. ~/LLM-integrated-Quantum-compiler
```

### 1.2 Pull the Open-WBO-Inc submodule

```bash
git submodule update --init --recursive
```

### 1.3 Install build dependencies (Ubuntu on Brev)

```bash
sudo apt-get update
sudo apt-get install -y build-essential make libgmp-dev
```

- `build-essential` → g++, etc.
- `make` → for the Makefile
- `libgmp-dev` → GMP (Open-WBO-Inc needs it)

### 1.4 Build the solver

```bash
cd lib/Open-WBO-Inc
make r
cd ../..
```

### 1.5 Verify

```bash
./lib/Open-WBO-Inc/open-wbo-inc_release --help
```

---

## 2. Python environment (satmapenv) setup

```bash
cd /path/to/LLM-integrated-Quantum-compiler   # your repo root

# Python 3.11 (install if missing)
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Create venv and install deps
python3.11 -m venv satmapenv
source satmapenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.1 Register as Jupyter kernel

To make the environment visible in Jupyter's kernel list (Notebook / Console):

```bash
cd /path/to/LLM-integrated-Quantum-compiler
source satmapenv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name satmapenv --display-name "Python (satmapenv)"
deactivate
```

Restart Jupyter (or refresh the launcher) and pick **"Python (satmapenv)"** from the kernel list.