"""
Hardware specification I/O for multi-qubit gate support.

Parses JSON hardware spec (edges + subgraph_matches per arity/type),
normalizes it, and exposes get_num_physical_qubits, get_edges,
get_subgraph_matches(spec, arity, type_name=None), and optionally
get_subgraph_match_types(spec, arity).

See docs/MULTI_QUBIT_IMPLEMENTATION_PLAN.md Section 2 and Step 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class HardwareSpecError(Exception):
    """Raised when the hardware spec is invalid or inconsistent."""
    pass


# ---------------------------------------------------------------------------
# Load and normalize
# ---------------------------------------------------------------------------

def load_spec(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and normalize the hardware spec from a file path or a dict.

    - If `source` is a path (str or Path), read JSON from file.
    - If `source` is a dict, use it as the raw spec (still normalized).
    - Validates: `edges` must be present.
    - Normalizes: computes num_physical_qubits if missing, validates indices,
      and derives subgraph_matches["2"] from edges if missing.

    Returns the normalized spec dict (may mutate the input dict if passed).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise HardwareSpecError(f"Hardware spec file not found: {path}")
        with open(path, "r") as f:
            spec = json.load(f)
    elif isinstance(source, dict):
        spec = source
    else:
        raise HardwareSpecError(
            "Hardware spec must be a file path (str or Path) or a dict"
        )

    if "edges" not in spec:
        raise HardwareSpecError("Hardware spec must contain 'edges'")

    edges = spec["edges"]
    subgraph_matches = spec.get("subgraph_matches") or {}

    # Infer num_physical_qubits if missing
    max_index = -1
    for u, v in edges:
        if not isinstance(u, int) or not isinstance(v, int):
            raise HardwareSpecError(
                f"Edge entries must be integers, got {type(u).__name__}, {type(v).__name__}"
            )
        max_index = max(max_index, u, v)
    for arity_key, types_dict in subgraph_matches.items():
        if not isinstance(types_dict, dict):
            raise HardwareSpecError(
                f"subgraph_matches['{arity_key}'] must be a dict of type -> list of tuples"
            )
        for _type_name, tuples_list in types_dict.items():
            for tup in tuples_list:
                for idx in tup:
                    if not isinstance(idx, int):
                        raise HardwareSpecError(
                            f"Subgraph tuple entries must be integers, got {type(idx).__name__}"
                        )
                    max_index = max(max_index, idx)

    if "num_physical_qubits" not in spec or spec["num_physical_qubits"] is None:
        spec["num_physical_qubits"] = max_index + 1
    num_phys = spec["num_physical_qubits"]

    # Validate indices in [0, num_phys - 1]
    def check_index(i: int, context: str) -> None:
        if i < 0 or i >= num_phys:
            raise HardwareSpecError(
                f"{context}: qubit index {i} out of range [0, {num_phys - 1}]"
            )

    for u, v in edges:
        check_index(u, "edges")
        check_index(v, "edges")
    for arity_key, types_dict in subgraph_matches.items():
        for type_name, tuples_list in types_dict.items():
            for tup in tuples_list:
                for i, idx in enumerate(tup):
                    check_index(idx, f"subgraph_matches['{arity_key}']['{type_name}']")

    # Derive subgraph_matches["2"] from edges if missing
    if "2" not in subgraph_matches or subgraph_matches["2"] is None:
        edge_tuples = []
        for [u, v] in edges:
            edge_tuples.append([u, v])
            edge_tuples.append([v, u])
        if "subgraph_matches" not in spec:
            spec["subgraph_matches"] = {}
        spec["subgraph_matches"]["2"] = {"edge": edge_tuples}

    return spec


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def get_num_physical_qubits(spec: Dict[str, Any]) -> int:
    """Return the number of physical qubits (after normalization)."""
    return spec["num_physical_qubits"]


def get_edges(spec: Dict[str, Any]) -> List[List[int]]:
    """
    Return the list of edges as [[u,v], ...].
    Used for SWAP legality and for building the coupling matrix.
    """
    return list(spec["edges"])


def get_subgraph_match_types(spec: Dict[str, Any], arity: int) -> List[str]:
    """Return the list of type names for the given arity (e.g. ['edge'], ['triangle','line3'])."""
    sub = spec.get("subgraph_matches") or {}
    by_arity = sub.get(str(arity))
    if by_arity is None:
        return []
    return list(by_arity.keys())


def get_subgraph_matches(
    spec: Dict[str, Any],
    arity: int,
    type_name: Optional[str] = None,
) -> List[List[int]]:
    """
    Return the list of allowed ordered n-tuples for the given arity (and optionally type).

    - If type_name is None, return the union of all types for that arity.
    - Each tuple is a list of n physical qubit indices: [p1, p2, ..., pn]
      meaning logical q1->p1, q2->p2, ..., qn->pn.
    - For arity 2, if the spec had no subgraph_matches["2"], load_spec() will
      have derived it from edges (both [u,v] and [v,u] per edge).
    """
    sub = spec.get("subgraph_matches") or {}
    by_arity = sub.get(str(arity))
    if by_arity is None:
        return []

    if type_name is not None:
        tuples_list = by_arity.get(type_name)
        if tuples_list is None:
            return []
        return [list(t) for t in tuples_list]

    # Union of all types for this arity
    result: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _tname, tuples_list in by_arity.items():
        for t in tuples_list:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                result.append(list(t))
    return result


def build_cm_from_spec(spec: Dict[str, Any]) -> "np.ndarray":
    """
    Build a coupling matrix (adjacency matrix) from the hardware spec edges.
    Returns numpy 2D array of shape (num_physical_qubits, num_physical_qubits)
    with 1 where an edge exists, 0 otherwise. Used by satmap_core for SWAP
    legality and by existing functions that expect cm.
    """
    n = get_num_physical_qubits(spec)
    cm = np.zeros((n, n), dtype=int)
    for (u, v) in get_edges(spec):
        cm[u, v] = 1
        cm[v, u] = 1
    return cm
