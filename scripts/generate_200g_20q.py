#!/usr/bin/env python3
"""Generate a 200-gate circuit on 20 qubits with varying qubit order (1q and 2q on many different qubits/pairs)."""
import random
import sys
from pathlib import Path

# Add src for qiskit (or use system qiskit)
import qiskit
from qiskit import QuantumCircuit

def main():
    n_qubits = 20
    n_gates = 200
    # Use fixed seed for reproducibility
    rng = random.Random(42)
    # 2q gates on random logical qubit pairs (varying qubit order; compiler must route to device edges)
    one_q_ops = ["h", "x", "y", "z", "t", "tdg", "s", "sdg"]
    circ = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        if rng.random() < 0.65:
            # 2-qubit gate: random distinct qubit pair (varying order across the device)
            c, t = rng.sample(range(n_qubits), 2)
            circ.cx(c, t)
        else:
            # 1-qubit gate on a random qubit
            q = rng.randint(0, n_qubits - 1)
            op = rng.choice(one_q_ops)
            getattr(circ, op)(q)
    out_dir = Path(__file__).resolve().parent.parent / "aux_files"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "circuit_200g_20q.qasm"
    with open(out_path, "w") as f:
        f.write(circ.qasm())
    print(f"Wrote {out_path} ({n_gates} gates, {n_qubits} qubits)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
