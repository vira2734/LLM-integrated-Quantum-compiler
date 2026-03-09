#!/usr/bin/env python3
"""Generate ~200 gates on 20 qubits: 1q, 2q, 3q (ccx), 4q (custom4), 5q (custom5) with varying qubit order."""
import random
import sys
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

def main():
    n_qubits = 20
    rng = random.Random(42)
    one_q_ops = ["h", "x", "y", "z", "t", "tdg", "s", "sdg"]

    # Custom 4- and 5-qubit gates (identity-like for structure)
    custom4 = Gate("custom4", 4, [])
    custom5 = Gate("custom5", 5, [])

    circ = QuantumCircuit(n_qubits)
    gate_count = 0
    target = 200
    while gate_count < target:
        r = rng.random()
        if r < 0.35:
            # 1q
            q = rng.randint(0, n_qubits - 1)
            getattr(circ, rng.choice(one_q_ops))(q)
            gate_count += 1
        elif r < 0.65:
            # 2q
            c, t = rng.sample(range(n_qubits), 2)
            circ.cx(c, t)
            gate_count += 1
        elif r < 0.82:
            # 3q (ccx / Toffoli)
            qubits = rng.sample(range(n_qubits), 3)
            circ.ccx(qubits[0], qubits[1], qubits[2])
            gate_count += 1
        elif r < 0.92:
            # 4q
            if n_qubits >= 4:
                qubits = rng.sample(range(n_qubits), 4)
                circ.append(custom4, qubits)
                gate_count += 1
        else:
            # 5q
            if n_qubits >= 5:
                qubits = rng.sample(range(n_qubits), 5)
                circ.append(custom5, qubits)
                gate_count += 1

    out_dir = Path(__file__).resolve().parent.parent / "aux_files"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "circuit_200g_20q_multi.qasm"
    with open(out_path, "w") as f:
        f.write(circ.qasm())
    print(f"Wrote {out_path}: {gate_count} gates on {n_qubits} qubits")
    return 0

if __name__ == "__main__":
    sys.exit(main())
