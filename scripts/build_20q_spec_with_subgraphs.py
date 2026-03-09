#!/usr/bin/env python3
"""Build 20-qubit hardware spec with subgraph_matches for arity 2, 3, 4, 5 (grid 4x5)."""
import json
from pathlib import Path

# 4x5 grid: row i has qubits [i*5, i*5+1, ..., i*5+4]
def grid_edges():
    edges = []
    for r in range(4):
        for c in range(5):
            n = r * 5 + c
            if c < 4:
                edges.append([n, n + 1])
            if r < 3:
                edges.append([n, n + 5])
    return edges

def both_orders(edges):
    out = []
    for u, v in edges:
        out.append([u, v])
        out.append([v, u])
    return out

def main():
    edges = grid_edges()
    # Arity 2: from edges (both orders)
    sm2_edge = both_orders(edges)

    # Arity 3: triangles (3 nodes that form a triangle on grid) and line3 (3 in a row or column)
    triangles = []
    for r in range(3):
        for c in range(4):
            # 2x2 square has 2 triangles
            a = r * 5 + c
            b = a + 1
            d = a + 5
            e = a + 6
            triangles.append([a, b, d])
            triangles.append([b, e, d])
    line3_h = []  # horizontal lines of 3
    for r in range(4):
        for c in range(3):
            a = r * 5 + c
            line3_h.append([a, a + 1, a + 2])
    line3_v = []  # vertical lines of 3
    for r in range(2):
        for c in range(5):
            a = r * 5 + c
            line3_v.append([a, a + 5, a + 10])
    sm3 = {"triangle": triangles, "line3": line3_h + line3_v}

    # Arity 4: line4 (row of 4 or col of 4), square (2x2)
    line4_h = []
    for r in range(4):
        for c in range(2):
            a = r * 5 + c
            line4_h.append([a, a + 1, a + 2, a + 3])
    line4_v = []
    for r in range(1):
        for c in range(5):
            a = r * 5 + c
            line4_v.append([a, a + 5, a + 10, a + 15])
    squares = []
    for r in range(3):
        for c in range(4):
            a = r * 5 + c
            squares.append([a, a + 1, a + 5, a + 6])
    sm4 = {"line4": line4_h + line4_v, "square": squares}

    # Arity 5: line5 (full row or full column)
    line5_rows = [[r * 5 + c for c in range(5)] for r in range(4)]
    line5_cols = [[r * 5 + c for r in range(4)] for c in range(5)]
    sm5 = {"line5": line5_rows + line5_cols}

    spec = {
        "num_physical_qubits": 20,
        "edges": edges,
        "subgraph_matches": {
            "2": {"edge": sm2_edge},
            "3": sm3,
            "4": sm4,
            "5": sm5,
        },
    }
    out = Path(__file__).resolve().parent.parent / "aux_files" / "hw_spec_20q_subgraphs.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote {out}")
    print(f"  edges: {len(edges)}, 2q: {len(sm2_edge)}, 3q types: triangle={len(triangles)} line3={len(line3_h)+len(line3_v)}, 4q: line4 square, 5q: line5")
    return 0

if __name__ == "__main__":
    exit(main())
