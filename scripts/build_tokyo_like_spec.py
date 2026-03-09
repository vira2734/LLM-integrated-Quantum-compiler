#!/usr/bin/env python3
"""Build Tokyo-sized (20 qubit) architecture JSON with subgraph_matches for 2, 3, 4, 5."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import architectures

def main():
    cm = architectures.ibmTokyo
    n = 20
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if cm[i, j] > 0:
                edges.append([i, j])

    def both_orders(edgelist):
        out = []
        for u, v in edgelist:
            out.append([u, v])
            out.append([v, u])
        return out

    sm2 = {"edge": both_orders(edges)}

    # Triangles: find all 3-cliques (triangles) in the graph
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[i] & adj[j]:
                    if k > j:
                        triangles.append([i, j, k])

    # Line3: consecutive in rows (0,1,2), (1,2,3), (2,3,4), (5,6,7), ... and cols (0,5,10), (1,6,11), ...
    line3 = []
    for row_start in [0, 5, 10, 15]:
        for c in range(3):
            line3.append([row_start + c, row_start + c + 1, row_start + c + 2])
    for col in range(5):
        for r in range(2):
            a, b, c = col + r * 5, col + (r + 1) * 5, col + (r + 2) * 5
            line3.append([a, b, c])
    sm3 = {"triangle": triangles[:80], "line3": line3}  # cap triangles for size

    # Line4: rows of 4, one column of 4
    line4 = []
    for row_start in [0, 5, 10, 15]:
        line4.append([row_start + c for c in range(4)])
    for col in range(5):
        line4.append([col + r * 5 for r in range(4)])
    # Square: 2x2 blocks (0,1,5,6), (1,2,6,7), (2,3,7,8), (3,4,8,9), (5,6,10,11), ...
    squares = []
    for row in range(3):
        for col in range(4):
            a = row * 5 + col
            squares.append([a, a + 1, a + 5, a + 6])
    sm4 = {"line4": line4, "square": squares}

    # Line5: full rows and full columns
    line5 = [[r * 5 + c for c in range(5)] for r in range(4)]
    line5 += [[c + r * 5 for r in range(4)] for c in range(5)]
    sm5 = {"line5": line5}

    spec = {
        "num_physical_qubits": 20,
        "edges": edges,
        "subgraph_matches": {
            "2": sm2,
            "3": sm3,
            "4": sm4,
            "5": sm5,
        },
    }
    out = Path(__file__).resolve().parent.parent / "examples" / "arch_tokyo_like_20q.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote {out}: 20 qubits, {len(edges)} edges, subgraph_matches 2/3/4/5")
    return 0

if __name__ == "__main__":
    exit(main())
