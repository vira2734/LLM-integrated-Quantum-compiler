import qiskit
from typing import Any, List, Optional, Union

def compose_swaps(swap_seq, phys_qubits):
    current = {phys : phys for phys in phys_qubits}
    for swap in swap_seq:
        apply_swap(swap, current)
    return current

def apply_swap(swap, current):
    [u, v] = swap
    for i in current.keys():
        if current[i] == u:
            current[i] = v
        elif current[i] == v:
            current[i] = u


def _qubits_from_gate(gate: Union[dict, list]) -> Any:
    """Return iterable of qubit indices for one gate (dict with 'qubits' or list/tuple)."""
    if isinstance(gate, dict) and "qubits" in gate:
        return gate["qubits"]
    return gate


def extract_qubits(gate_list: List[Union[dict, list]]) -> set:
    """
    Return the set of logical qubit indices used in gate_list.
    gate_list can be from extract2qubit (list of [c, t] lists) or from extract_gates (list of dicts).
    """
    qubits = set()
    for gate in gate_list:
        for qubit in _qubits_from_gate(gate):
            qubits.add(qubit)
    return qubits


def extract_gates(fname: str) -> List[dict]:
    """
    Parse QASM file into a list of gate entries. Preserves program order.
    Each entry: {"name": str, "qubits": (q0, q1, ...), "arity": int, "type": Optional[str]}.
    Supports 1-, 2-, and n-qubit gates (including custom gates from QASM).
    """
    gates = []
    circ = qiskit.QuantumCircuit.from_qasm_file(fname)
    for j in range(len(circ)):
        op = circ[j][0]
        qubit_objs = circ[j][1]
        name = op.name if hasattr(op, "name") else str(op)
        indices = tuple(circ.find_bit(q)[0] for q in qubit_objs)
        n = len(indices)
        arity = n
        # type: leave None for default (union of all types for that arity in hardware spec)
        gates.append({
            "name": name,
            "qubits": indices,
            "arity": arity,
            "type": None,
        })
    return gates


def extract2qubit(fname: str) -> List[List[int]]:
    """
    Legacy: return list of 2-qubit gates only, each [control, target].
    Gates with != 2 qubits are skipped (with a warning for >2).
    """
    gates = []
    circ = qiskit.QuantumCircuit.from_qasm_file(fname)
    for j in range(len(circ)):
        qubits = circ[j][1]
        if len(qubits) == 2:
            gates.append([circ.find_bit(q)[0] for q in qubits])
        elif len(qubits) > 2:
            print('Warning: ignoring gate with more than 2 qubits')
    return gates


# ---------------------------------------------------------------------------
# Step 3: Multi-qubit gate indices and chunking (k = multi-qubit gate index)
# ---------------------------------------------------------------------------

def get_multi_qubit_indices(gate_list: List[dict]) -> List[int]:
    """
    Return indices into gate_list where the gate has arity >= 2 (multi-qubit).
    Step k in the encoding refers to the k-th multi-qubit gate (0-indexed).
    """
    return [i for i, g in enumerate(gate_list) if g.get("arity", 0) >= 2]


def get_chunk_ranges(
    gate_list: List[dict],
    num_chunks: int,
) -> List[tuple]:
    """
    Split the gate list into chunks by multi-qubit gate count.
    Returns list of (start_idx, end_idx_excl, currentSize) for each chunk.
    - start_idx, end_idx_excl: slice gate_list[start_idx:end_idx_excl] is this chunk.
    - currentSize: number of multi-qubit gates in this chunk (k runs 0 .. currentSize-1).
    Single-qubit gates between multi-qubit gates belong to the chunk of the
    multi-qubit gate that follows them.
    """
    mq_indices = get_multi_qubit_indices(gate_list)
    num_multi = len(mq_indices)
    if num_multi == 0:
        return []
    if num_chunks <= 0:
        num_chunks = 1
    if num_chunks >= num_multi:
        num_chunks = num_multi
    # Even split: first (num_multi % num_chunks) chunks get +1
    base_size = num_multi // num_chunks
    remainder = num_multi % num_chunks
    sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]
    ranges = []
    pos = 0
    for i in range(num_chunks):
        current_size = sizes[i]
        if current_size == 0:
            continue
        # Chunk 0 starts at gate index 0; chunk i > 0 starts right after last gate of chunk i-1.
        start_idx = 0 if i == 0 else mq_indices[pos - 1] + 1
        end_mq_idx = mq_indices[pos + current_size - 1]
        end_idx_excl = end_mq_idx + 1
        ranges.append((start_idx, end_idx_excl, current_size))
        pos += current_size
    return ranges
