#!/usr/bin/env python3
"""
Build target hardware spec (arch JSON with subgraph_matches) from:
  1. Raw hardware map (num_physical_qubits + edges only)
  2. LLM output: {"rules": [{"nQubits": int, "shape": str, "edges": [[int,int],...]}, ...]}

Uses subgraph isomorphism (NetworkX VF2) to find all embeddings of each rule's
pattern graph in the hardware graph. Arity-2 matches are always derived from
edges (both directions). Output format is compatible with hardware_spec.load_spec()
and satmap's -H/--hardware_spec.

Requires: pip install networkx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
except ImportError:
    print("This script requires networkx. Install with: pip install networkx", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Types (LLM output and spec)
# ---------------------------------------------------------------------------

def load_raw_hardware(path: Path) -> Dict[str, Any]:
    """Load raw hardware JSON: must have 'edges'; may have 'num_physical_qubits'."""
    with open(path, "r") as f:
        spec = json.load(f)
    if "edges" not in spec:
        raise ValueError(f"{path}: missing 'edges'")
    edges = spec["edges"]
    n = spec.get("num_physical_qubits")
    if n is None:
        max_idx = max(u for e in edges for u in e) if edges else -1
        n = max_idx + 1
    spec["num_physical_qubits"] = n
    return spec


def load_llm_rules(source: Path | str | Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load LLM output: {'rules': [{'nQubits': int, 'shape': str, 'edges': [[int,int],...]}, ...]}."""
    if isinstance(source, dict):
        data = source
    else:
        with open(source, "r") as f:
            data = json.load(f)
    if "rules" not in data:
        raise ValueError("LLM output must contain 'rules' key")
    return data["rules"]


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def edges_to_undirected_graph(edges: List[List[int]]) -> "nx.Graph":
    """Build an undirected NetworkX graph from a list of [u, v] edges."""
    G = nx.Graph()
    for (u, v) in edges:
        G.add_edge(u, v)
    return G


def rule_to_pattern_graph(rule: Dict[str, Any]) -> "nx.Graph":
    """Build undirected pattern graph from one LLM rule. Nodes are 0..nQubits-1."""
    n = rule["nQubits"]
    edge_list = rule.get("edges", [])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (u, v) in edge_list:
        if 0 <= u < n and 0 <= v < n:
            G.add_edge(u, v)
    return G


def type_name_for_rule(rule: Dict[str, Any]) -> str:
    """Map rule shape + nQubits to arch type name (e.g. 'line' + 3 -> 'line3')."""
    shape = (rule.get("shape") or "line").strip().lower()
    n = rule["nQubits"]
    if shape == "line":
        return f"line{n}"
    return shape


# ---------------------------------------------------------------------------
# Subgraph matching (induced subgraph isomorphism)
# ---------------------------------------------------------------------------

def find_subgraph_matches(
    hardware_graph: "nx.Graph",
    pattern_graph: "nx.Graph",
    n_qubits: int,
) -> List[List[int]]:
    """
    Find all induced subgraph isomorphisms from pattern to hardware.
    Returns list of ordered tuples [p0, p1, ..., p_{n-1}] (physical qubit indices).
    Uses NetworkX VF2; each mapping preserves structure (no extra edges in image).
    """
    if pattern_graph.number_of_nodes() != n_qubits or pattern_graph.number_of_nodes() == 0:
        return []

    # GraphMatcher(G1, G2): finds subgraphs of G1 isomorphic to G2.
    # So we want: hardware = G1, pattern = G2 -> mappings from pattern nodes to hardware nodes.
    matcher = nx.algorithms.isomorphism.GraphMatcher(hardware_graph, pattern_graph)
    results: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()

    for mapping in matcher.subgraph_isomorphisms_iter():
        # GraphMatcher(H, P) yields mapping: hardware_node -> pattern_node (H -> P).
        # We need ordered tuple (physical for 0, physical for 1, ...) = inverse mapping.
        inv = {v: k for k, v in mapping.items()}
        if len(inv) != n_qubits:
            continue
        ordered = [inv[i] for i in range(n_qubits)]
        key = tuple(ordered)
        if key in seen:
            continue
        # Induced: image of P in H has no extra edges. Mapping is H->P so inv is P->H.
        if _is_induced(hardware_graph, pattern_graph, inv):
            seen.add(key)
            results.append(ordered)
    return results


def _is_induced(
    H: "nx.Graph",
    P: "nx.Graph",
    mapping: Dict[int, int],
) -> bool:
    """True iff the image of P under mapping is an induced subgraph of H (no extra edges)."""
    for i in P.nodes():
        for j in P.nodes():
            if i >= j:
                continue
            hi, hj = mapping[i], mapping[j]
            in_P = P.has_edge(i, j)
            in_H = H.has_edge(hi, hj)
            if in_H and not in_P:
                return False
    return True


# ---------------------------------------------------------------------------
# Arity-2: all edges in both directions
# ---------------------------------------------------------------------------

def derive_arity2_edge_tuples(edges: List[List[int]]) -> List[List[int]]:
    """Return [[u,v], [v,u], ...] for each edge [u,v]. Used for subgraph_matches['2']['edge']."""
    out: List[List[int]] = []
    for (u, v) in edges:
        out.append([u, v])
        out.append([v, u])
    return out


# ---------------------------------------------------------------------------
# Build target spec
# ---------------------------------------------------------------------------

def build_target_spec(
    raw_spec: Dict[str, Any],
    llm_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build full hardware spec with subgraph_matches from raw spec and LLM rules.
    - Arity 2 is always set from raw edges (both directions).
    - For each rule with nQubits >= 2, find all pattern matches and add to subgraph_matches[str(nQubits)][type_name].
    """
    edges = raw_spec["edges"]
    n_phys = raw_spec["num_physical_qubits"]
    H = edges_to_undirected_graph(edges)

    subgraph_matches: Dict[str, Dict[str, List[List[int]]]] = {}
    # Arity 2: default from edges
    subgraph_matches["2"] = {"edge": derive_arity2_edge_tuples(edges)}

    for rule in llm_rules:
        n = rule["nQubits"]
        if n < 2:
            continue
        if n == 2:
            # Already set above
            continue
        shape_edges = rule.get("edges", [])
        if not shape_edges:
            continue
        P = rule_to_pattern_graph(rule)
        if P.number_of_nodes() != n:
            continue
        type_name = type_name_for_rule(rule)
        matches = find_subgraph_matches(H, P, n)
        if not matches:
            continue
        arity_key = str(n)
        if arity_key not in subgraph_matches:
            subgraph_matches[arity_key] = {}
        # If same type_name appears in multiple rules (e.g. two "line" rules), merge
        if type_name in subgraph_matches[arity_key]:
            existing = {tuple(t) for t in subgraph_matches[arity_key][type_name]}
            for m in matches:
                if tuple(m) not in existing:
                    existing.add(tuple(m))
                    subgraph_matches[arity_key][type_name].append(m)
        else:
            subgraph_matches[arity_key][type_name] = matches

    return {
        "num_physical_qubits": n_phys,
        "edges": edges,
        "subgraph_matches": subgraph_matches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build target hardware spec (subgraph_matches) from raw edges JSON and LLM rules JSON.",
    )
    parser.add_argument(
        "raw_hardware",
        type=Path,
        help="Path to raw hardware JSON (num_physical_qubits + edges)",
    )
    parser.add_argument(
        "llm_rules",
        type=Path,
        help="Path to LLM output JSON: {\"rules\": [{\"nQubits\", \"shape\", \"edges\"}, ...]}",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for target arch JSON (default: stdout)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2)",
    )
    args = parser.parse_args()

    raw_spec = load_raw_hardware(args.raw_hardware)
    llm_rules = load_llm_rules(args.llm_rules)
    target = build_target_spec(raw_spec, llm_rules)

    out_json = json.dumps(target, indent=args.indent)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
