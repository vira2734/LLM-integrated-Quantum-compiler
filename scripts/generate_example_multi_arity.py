#!/usr/bin/env python3
"""Generate ~220-gate example (like 4_49_16) with 1q, 2q, 3q, 4q, 5q gates on 16 qubits."""
import random
import sys
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

def main():
    n_qubits = 16
    target_gates = 218  # same ballpark as 4_49_16.qasm
    rng = random.Random(42)
    one_q_ops = ["h", "t", "tdg", "x", "z"]
    custom4 = Gate("custom4", 4, [])
    custom5 = Gate("custom5", 5, [])

    circ = QuantumCircuit(n_qubits)
    gate_count = 0
    while gate_count < target_gates:
        r = rng.random()
        if r < 0.40:
            q = rng.randint(0, n_qubits - 1)
            getattr(circ, rng.choice(one_q_ops))(q)
            gate_count += 1
        elif r < 0.78:
            c, t = rng.sample(range(n_qubits), 2)
            circ.cx(c, t)
            gate_count += 1
        elif r < 0.90:
            qubits = rng.sample(range(n_qubits), 3)
            circ.ccx(qubits[0], qubits[1], qubits[2])
            gate_count += 1
        elif r < 0.96 and gate_count < target_gates:
            qubits = rng.sample(range(n_qubits), 4)
            circ.append(custom4, qubits)
            gate_count += 1
        elif gate_count < target_gates:
            qubits = rng.sample(range(n_qubits), 5)
            circ.append(custom5, qubits)
            gate_count += 1

    out = Path(__file__).resolve().parent.parent / "examples" / "multi_arity_16q.qasm"
    with open(out, "w") as f:
        f.write(circ.qasm())
    print(f"Wrote {out}: {gate_count} gates on {n_qubits} qubits (1q/2q/3q/4q/5q)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
