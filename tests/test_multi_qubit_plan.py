"""
Tests for the multi-qubit implementation plan.

Run from project root (LLM-SAT-compiler) with:
  source satmapenv/bin/activate && PYTHONPATH=src python -m unittest tests.test_multi_qubit_plan -v

Fixtures: tests/fixtures/ (hw_spec_rich.json, circuit_*.qasm).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# Ensure src is on path when running tests from project root
import sys
_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import hardware_spec as hw

# Fixtures path (relative to project root when running from project root)
FIXTURES_DIR = _here / "fixtures"


def _open_wbo_available():
    """True if Open-WBO-Inc solver binary exists (for integration/e2e tests)."""
    # solve() uses path relative to cwd: lib/Open-WBO-Inc/open-wbo-inc_release
    for root in (_here.parent, Path.cwd()):
        exe = root / "lib" / "Open-WBO-Inc" / "open-wbo-inc_release"
        if exe.is_file() and os.access(exe, os.X_OK):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 1: Hardware Spec I/O
# ---------------------------------------------------------------------------

class TestStep1HardwareSpec(unittest.TestCase):
    """Step 1: Hardware Spec I/O tests."""

    def test_parse_valid_spec(self):
        """Load a valid JSON with edges and subgraph_matches; assert no exception and expected values."""
        spec_dict = {
            "num_physical_qubits": 20,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "subgraph_matches": {
                "2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]},
                "3": {"triangle": [[0, 1, 2], [1, 2, 3]]},
            },
        }
        spec = hw.load_spec(spec_dict)
        self.assertEqual(hw.get_num_physical_qubits(spec), 20)
        edges = hw.get_edges(spec)
        self.assertEqual(len(edges), 3)
        self.assertIn([0, 1], edges)
        self.assertIn([1, 2], edges)
        self.assertIn([2, 3], edges)

    def test_parse_valid_spec_from_file(self):
        """Same as test_parse_valid_spec but load from a temp file."""
        spec_dict = {
            "num_physical_qubits": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "subgraph_matches": {"2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(spec_dict, f)
            path = f.name
        try:
            spec = hw.load_spec(path)
            self.assertEqual(hw.get_num_physical_qubits(spec), 4)
            self.assertEqual(len(hw.get_edges(spec)), 3)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_missing_num_qubits(self):
        """Omit num_physical_qubits; assert it is inferred correctly from edges and subgraph_matches."""
        spec_dict = {
            "edges": [[0, 1], [1, 2], [5, 6]],
            "subgraph_matches": {
                "3": {"line3": [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]},
            },
        }
        spec = hw.load_spec(spec_dict)
        self.assertEqual(hw.get_num_physical_qubits(spec), 7)

    def test_subgraph_matches_arity_2(self):
        """For arity 2, request type 'edge'; assert returned list equals expected (both orders per edge)."""
        spec_dict = {
            "num_physical_qubits": 4,
            "edges": [[0, 1], [1, 2]],
            "subgraph_matches": {
                "2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1]]},
            },
        }
        spec = hw.load_spec(spec_dict)
        matches = hw.get_subgraph_matches(spec, 2, type_name="edge")
        self.assertEqual(len(matches), 4)
        self.assertIn([0, 1], matches)
        self.assertIn([1, 0], matches)
        self.assertIn([1, 2], matches)
        self.assertIn([2, 1], matches)

    def test_subgraph_matches_arity_3(self):
        """For arity 3, request type 'triangle'; assert equals JSON. Request type None; assert union."""
        spec_dict = {
            "num_physical_qubits": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "subgraph_matches": {
                "3": {
                    "triangle": [[0, 1, 2], [1, 2, 3]],
                    "line3": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                },
            },
        }
        spec = hw.load_spec(spec_dict)
        triangle_only = hw.get_subgraph_matches(spec, 3, type_name="triangle")
        self.assertEqual(triangle_only, [[0, 1, 2], [1, 2, 3]])
        union = hw.get_subgraph_matches(spec, 3, type_name=None)
        # triangle: (0,1,2), (1,2,3); line3: (0,1,2), (1,2,3), (2,3,4) -> 3 unique
        self.assertEqual(len(union), 3)
        self.assertIn([0, 1, 2], union)
        self.assertIn([1, 2, 3], union)
        self.assertIn([2, 3, 4], union)

    def test_2_qubit_derived_from_edges(self):
        """Spec with only edges, no subgraph_matches['2']; get_subgraph_matches(spec, 2) derived from edges."""
        spec_dict = {"edges": [[0, 1], [1, 2]]}
        spec = hw.load_spec(spec_dict)
        self.assertEqual(hw.get_num_physical_qubits(spec), 3)
        matches = hw.get_subgraph_matches(spec, 2)
        self.assertEqual(len(matches), 4)
        self.assertIn([0, 1], matches)
        self.assertIn([1, 0], matches)
        self.assertIn([1, 2], matches)
        self.assertIn([2, 1], matches)
        matches_edge = hw.get_subgraph_matches(spec, 2, type_name="edge")
        self.assertEqual(set(tuple(m) for m in matches_edge), set(tuple(m) for m in matches))

    def test_invalid_index(self):
        """Spec with edge [0, 100] and num_physical_qubits 20; assert normalize raises clear error."""
        spec_dict = {"num_physical_qubits": 20, "edges": [[0, 100]]}
        with self.assertRaises(hw.HardwareSpecError) as ctx:
            hw.load_spec(spec_dict)
        msg = str(ctx.exception)
        self.assertIn("100", msg)
        self.assertIn("range", msg.lower())

    def test_missing_edges(self):
        """Spec without 'edges' key must raise."""
        with self.assertRaises(hw.HardwareSpecError) as ctx:
            hw.load_spec({"num_physical_qubits": 4})
        self.assertIn("edges", str(ctx.exception).lower())

    def test_get_subgraph_match_types(self):
        """Optional API: get_subgraph_match_types(spec, arity) returns type names."""
        spec_dict = {
            "num_physical_qubits": 5,
            "edges": [[0, 1], [1, 2]],
            "subgraph_matches": {
                "3": {"triangle": [[0, 1, 2]], "line3": [[0, 1, 2], [1, 2, 3]]},
            },
        }
        spec = hw.load_spec(spec_dict)
        types_3 = hw.get_subgraph_match_types(spec, 3)
        self.assertEqual(set(types_3), {"triangle", "line3"})
        types_2 = hw.get_subgraph_match_types(spec, 2)
        self.assertIn("edge", types_2)


# ---------------------------------------------------------------------------
# Step 2: Circuit Input — Generic Gate List
# ---------------------------------------------------------------------------

class TestStep2GateList(unittest.TestCase):
    """Step 2: extract_gates and gate list structure."""

    def _write_qasm(self, circ):
        """Write circuit to temp file, return path."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
            f.write(circ.qasm())
            return f.name

    def test_single_qubit_only(self):
        """QASM with only 1-qubit gates; gate list has correct length and each arity is 1."""
        import qiskit
        circ = qiskit.QuantumCircuit(2)
        circ.h(0)
        circ.x(1)
        circ.z(0)
        path = self._write_qasm(circ)
        try:
            from common import extract_gates
            gates = extract_gates(path)
            self.assertEqual(len(gates), 3)
            for g in gates:
                self.assertEqual(g["arity"], 1)
                self.assertEqual(len(g["qubits"]), 1)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_two_qubit_only(self):
        """QASM with only CX; gate list matches extract2qubit output (order and qubit indices)."""
        import qiskit
        circ = qiskit.QuantumCircuit(3)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 0)
        path = self._write_qasm(circ)
        try:
            from common import extract_gates, extract2qubit
            gates = extract_gates(path)
            cnots = extract2qubit(path)
            self.assertEqual(len(gates), 3)
            for i, g in enumerate(gates):
                self.assertEqual(g["arity"], 2)
                self.assertEqual(tuple(g["qubits"]), tuple(cnots[i]))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_mixed_1q_2q(self):
        """QASM with H, CX, X, CX; list has 4 entries, arities 1, 2, 1, 2 and qubit indices correct."""
        import qiskit
        circ = qiskit.QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.x(1)
        circ.cx(0, 1)
        path = self._write_qasm(circ)
        try:
            from common import extract_gates
            gates = extract_gates(path)
            self.assertEqual(len(gates), 4)
            self.assertEqual(gates[0]["arity"], 1)
            self.assertEqual(gates[0]["qubits"], (0,))
            self.assertEqual(gates[1]["arity"], 2)
            self.assertEqual(gates[1]["qubits"], (0, 1))
            self.assertEqual(gates[2]["arity"], 1)
            self.assertEqual(gates[2]["qubits"], (1,))
            self.assertEqual(gates[3]["arity"], 2)
            self.assertEqual(gates[3]["qubits"], (0, 1))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_three_qubit(self):
        """QASM containing a 3-qubit gate (e.g. ccx); one entry with arity 3 and three qubit indices."""
        import qiskit
        circ = qiskit.QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        path = self._write_qasm(circ)
        try:
            from common import extract_gates
            gates = extract_gates(path)
            self.assertGreaterEqual(len(gates), 1)
            ccx_gates = [g for g in gates if g["arity"] == 3]
            self.assertGreaterEqual(len(ccx_gates), 1)
            g = ccx_gates[0]
            self.assertEqual(len(g["qubits"]), 3)
            self.assertEqual(g["qubits"], (0, 1, 2))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_gate_order_preserved(self):
        """Order of gates in file vs list is identical."""
        import qiskit
        circ = qiskit.QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.x(0)
        circ.z(1)
        circ.cx(1, 0)
        path = self._write_qasm(circ)
        try:
            from common import extract_gates
            gates = extract_gates(path)
            self.assertEqual(len(gates), 5)
            self.assertEqual(gates[0]["name"], "h")
            self.assertEqual(gates[1]["name"], "cx")
            self.assertEqual(gates[2]["name"], "x")
            self.assertEqual(gates[3]["name"], "z")
            self.assertEqual(gates[4]["name"], "cx")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_qubits_with_gate_list(self):
        """extract_qubits works with list of gate dicts from extract_gates."""
        from common import extract_qubits
        gate_list = [
            {"name": "h", "qubits": (0,), "arity": 1, "type": None},
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
        ]
        qubits = extract_qubits(gate_list)
        self.assertEqual(qubits, {0, 1})

    def test_extract_qubits_with_legacy_list(self):
        """extract_qubits still works with legacy list of [c,t] from extract2qubit."""
        from common import extract_qubits
        gate_list = [[0, 1], [1, 2]]
        qubits = extract_qubits(gate_list)
        self.assertEqual(qubits, {0, 1, 2})


# ---------------------------------------------------------------------------
# Step 3: Unify gate index and time steps (chunking)
# ---------------------------------------------------------------------------

class TestStep3Chunking(unittest.TestCase):
    """Step 3: Multi-qubit indices and chunk boundaries."""

    def test_multi_qubit_indices(self):
        """Gate list with 2 single-qubit and 3 two-qubit; multi-qubit indices are [1, 3, 4]."""
        from common import get_multi_qubit_indices
        gate_list = [
            {"name": "h", "qubits": (0,), "arity": 1, "type": None},
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
            {"name": "x", "qubits": (1,), "arity": 1, "type": None},
            {"name": "cx", "qubits": (1, 0), "arity": 2, "type": None},
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
        ]
        indices = get_multi_qubit_indices(gate_list)
        self.assertEqual(indices, [1, 3, 4])

    def test_chunk_boundaries(self):
        """10 multi-qubit gates, 2 chunks; chunk 0 has 5, chunk 1 has 5."""
        from common import get_chunk_ranges, get_multi_qubit_indices
        # Build gate list with 10 two-qubit gates (and some single in between)
        gate_list = []
        for i in range(10):
            if i > 0:
                gate_list.append({"name": "x", "qubits": (0,), "arity": 1, "type": None})
            gate_list.append({"name": "cx", "qubits": (0, 1), "arity": 2, "type": None})
        self.assertEqual(len(get_multi_qubit_indices(gate_list)), 10)
        ranges = get_chunk_ranges(gate_list, 2)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][2], 5)  # currentSize
        self.assertEqual(ranges[1][2], 5)
        # Slice lengths: chunk 0 should have 5 multi-qubit gates in its slice
        slice0 = gate_list[ranges[0][0]:ranges[0][1]]
        mq0 = [g for g in slice0 if g["arity"] >= 2]
        self.assertEqual(len(mq0), 5)
        slice1 = gate_list[ranges[1][0]:ranges[1][1]]
        mq1 = [g for g in slice1 if g["arity"] >= 2]
        self.assertEqual(len(mq1), 5)

    def test_k_range(self):
        """For a chunk with currentSize 4, k runs 0..3 and corresponds to 4 multi-qubit gates."""
        from common import get_chunk_ranges
        gate_list = [
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
            {"name": "cx", "qubits": (1, 0), "arity": 2, "type": None},
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
            {"name": "cx", "qubits": (1, 0), "arity": 2, "type": None},
        ]
        ranges = get_chunk_ranges(gate_list, 1)  # one chunk
        self.assertEqual(len(ranges), 1)
        start, end_excl, current_size = ranges[0]
        self.assertEqual(current_size, 4)
        for k in range(current_size):
            self.assertLess(k, current_size)
        self.assertEqual(end_excl - start, 4)


# ---------------------------------------------------------------------------
# Step 4: New multi-qubit constraint (writeMultiQubitGateConstraint)
# Step 5: Variable layout (flatten/unravel with match_counts)
# ---------------------------------------------------------------------------

class TestStep4Step5ConstraintAndLayout(unittest.TestCase):
    """Step 4: Multi-qubit constraint. Step 5: Flatten/unravel with match variables."""

    def test_3q_one_match(self):
        """One 3-qubit gate, SubgraphMatches has one tuple; at least one match clause and implication clauses."""
        import hardware_spec as hw
        import satmap_core as sc
        spec = hw.load_spec({
            "num_physical_qubits": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
            "subgraph_matches": {
                "2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]},
                "3": {"triangle": [[0, 1, 2]]},
            },
        })
        multi_qubit_gates = [{"name": "ccx", "qubits": (0, 1, 2), "arity": 3, "type": "triangle"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp:
            path = tmp.name
        try:
            with open(path, "w") as f:
                match_counts = sc.writeMultiQubitGateConstraint(
                    multi_qubit_gates, spec, physNum=4, logNum=3, swapNum=1, top=1000, path=f
                )
            self.assertEqual(match_counts, [1])
            with open(path) as f:
                content = f.read()
            self.assertIn("1000", content)  # hard clause
        finally:
            Path(path).unlink(missing_ok=True)

    def test_3q_two_matches(self):
        """One 3-qubit gate, SubgraphMatches has two tuples; both appear in encoding."""
        import hardware_spec as hw
        import satmap_core as sc
        spec = hw.load_spec({
            "num_physical_qubits": 5,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "subgraph_matches": {
                "2": {"edge": [[0, 1], [1, 0]]},
                "3": {"line3": [[0, 1, 2], [1, 2, 3]]},
            },
        })
        multi_qubit_gates = [{"name": "ccx", "qubits": (0, 1, 2), "arity": 3, "type": None}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp:
            path = tmp.name
        try:
            with open(path, "w") as f:
                match_counts = sc.writeMultiQubitGateConstraint(
                    multi_qubit_gates, spec, physNum=5, logNum=3, swapNum=1, top=1000, path=f
                )
            self.assertEqual(match_counts, [2])  # union of line3 has 2 tuples
        finally:
            Path(path).unlink(missing_ok=True)

    def test_no_subgraph_matches(self):
        """Gate has arity 3 but spec has no subgraph_matches for 3; clear error at constraint time."""
        import hardware_spec as hw
        import satmap_core as sc
        spec = hw.load_spec({"edges": [[0, 1], [1, 2]], "subgraph_matches": {"2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1]]}}})
        multi_qubit_gates = [{"name": "ccx", "qubits": (0, 1, 2), "arity": 3, "type": None}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp:
            path = tmp.name
        try:
            with open(path, "w") as f:
                with self.assertRaises(ValueError) as ctx:
                    sc.writeMultiQubitGateConstraint(
                        multi_qubit_gates, spec, physNum=3, logNum=3, swapNum=1, top=1000, path=f
                    )
            self.assertIn("subgraph_matches", str(ctx.exception).lower())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_flatten_unravel_roundtrip(self):
        """For small (physNum, logNum, numGates, swapNum) with match_counts, flatten then unravel."""
        import satmap_core as sc
        physNum, logNum, numCnots, swapNum = 2, 2, 2, 1
        match_counts = [2, 2]  # 2 matches per gate
        # Map literal (False, "x", i, j, k)
        lit_x = (False, "x", 0, 0, 0)
        flat_x = sc.flattenedIndex(lit_x, physNum, logNum, numCnots, swapNum, match_counts)
        back = sc.unravel(flat_x, physNum, logNum, numCnots, swapNum, match_counts)
        self.assertEqual(back[0], lit_x[0])
        self.assertEqual(back[1], lit_x[1])
        self.assertEqual(tuple(back[2]), (0, 0, 0))
        # Match literal (False, "m", j, k)
        lit_m = (False, "m", 1, 1)
        flat_m = sc.flattenedIndex(lit_m, physNum, logNum, numCnots, swapNum, match_counts)
        back_m = sc.unravel(flat_m, physNum, logNum, numCnots, swapNum, match_counts)
        self.assertEqual(back_m[1], "m")
        self.assertEqual(back_m[2], (1, 1))
        # Negated
        lit_neg = (True, "x", 1, 1, 0)
        flat_neg = sc.flattenedIndex(lit_neg, physNum, logNum, numCnots, swapNum, match_counts)
        back_neg = sc.unravel(flat_neg, physNum, logNum, numCnots, swapNum, match_counts)
        self.assertTrue(back_neg[0])
        self.assertEqual(back_neg[1], "x")

    def test_2q_constraint_equivalent(self):
        """For 2-qubit-only gates and edges-only spec, new constraint uses map + match; legacy uses p/r/x. Both produce clauses."""
        import satmap_core as sc
        import numpy as np
        # One CNOT, two edges (0,1) and (1,0) -> 2 matches for 2q
        spec = hw.load_spec({"num_physical_qubits": 2, "edges": [[0, 1]], "subgraph_matches": {"2": {"edge": [[0, 1], [1, 0]]}}})
        cm = hw.build_cm_from_spec(spec)
        cnots = [(0, 1)]
        multi_qubit = [{"name": "cx", "qubits": (0, 1), "arity": 2, "type": None}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp_new:
            path_new = tmp_new.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp_legacy:
            path_legacy = tmp_legacy.name
        try:
            with open(path_new, "w") as f:
                match_counts = sc.writeMultiQubitGateConstraint(
                    multi_qubit, spec, physNum=2, logNum=2, swapNum=1, top=1000, path=f
                )
            self.assertEqual(match_counts, [2])
            with open(path_new) as f:
                new_content = f.read()
            self.assertIn("1000", new_content)
            # At least one match clause (2 m literals) + 2*2 implication clauses = 5 clauses
            self.assertGreaterEqual(new_content.count("0\n"), 5)
            # Legacy: write only writeCnotConstraint to a file (same cm)
            with open(path_legacy, "w") as f:
                sc.writeCnotConstraint(cnots, cm, physNum=2, logNum=2, swapNum=1, top=1000, path=f)
            with open(path_legacy) as f:
                legacy_content = f.read()
            self.assertIn("1000", legacy_content)
            self.assertGreater(len(legacy_content), 0)
        finally:
            Path(path_new).unlink(missing_ok=True)
            Path(path_legacy).unlink(missing_ok=True)

    def test_variable_count(self):
        """After generating all constraints for a tiny instance, all variable indices in [1, total_vars]."""
        import satmap_core as sc
        physNum, logNum, numCnots, swapNum = 2, 2, 2, 1
        spec = hw.load_spec({"num_physical_qubits": 2, "edges": [[0, 1]], "subgraph_matches": {"2": {"edge": [[0, 1], [1, 0]]}}})
        cm = hw.build_cm_from_spec(spec)
        multi_qubit = [
            {"name": "cx", "qubits": (0, 1), "arity": 2, "type": None},
            {"name": "cx", "qubits": (1, 0), "arity": 2, "type": None},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wcnf", delete=False) as tmp:
            path = tmp.name
        try:
            s, match_counts = sc.generateAndWriteClauses(
                logNum, [], [], cm, swapNum, [], path,
                routing=True, spec=spec, multi_qubit_gates=multi_qubit
            )
            self.assertIsNotNone(match_counts)
            numP = physNum * physNum * numCnots
            numR = numP
            numX = numCnots * logNum * physNum
            numS = numCnots * physNum * physNum * swapNum
            numB = numS
            numW = physNum * physNum * numCnots
            numD = logNum * numCnots
            numM = sum(match_counts)
            total_vars = numP + numR + numX + numS + numB + numW + numD + numM
            seen_vars = set()
            with open(path) as f:
                for line in f:
                    if line.startswith("p "):
                        continue
                    parts = line.split()
                    for p in parts:
                        if p == "0":
                            break
                        try:
                            v = abs(int(p))
                            if v > 0:
                                seen_vars.add(v)
                        except ValueError:
                            pass
            for v in seen_vars:
                self.assertGreaterEqual(v, 1, f"Variable {v} < 1")
                self.assertLessEqual(v, total_vars, f"Variable {v} > total_vars {total_vars}")
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 6: Solving (integration; require Open-WBO-Inc unless skipped)
# ---------------------------------------------------------------------------

class TestStep6Solving(unittest.TestCase):
    """Step 6: Full solve with spec; skip if Open-WBO-Inc not available."""

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_solve_small_2q(self):
        """Full pipeline on small 2-qubit-only circuit + hardware spec; assert sat and solution returned."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_2q_only.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "prob")
            sname = os.path.join(td, "sol")
            # Copy circuit to aux-like path for solve (solve writes qiskit-* and reads from progPath)
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            prog_use = qasm_dest
            results = sc.solve(prog_use, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=30)
        self.assertIn("swaps", results)
        self.assertIn("gate_list", results)
        self.assertIn("match_counts_per_chunk", results)
        self.assertEqual(len(results["gate_list"]), 2)
        self.assertEqual(len(results["match_counts_per_chunk"]), 1)

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_solve_small_3q(self):
        """Solve circuit with one 3-qubit gate + spec with subgraph_matches for 3; assert sat."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_2q_3q.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "prob")
            sname = os.path.join(td, "sol")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=45)
        self.assertIn("swaps", results)
        self.assertIn("match_counts_per_chunk", results)
        # 3 multi-qubit gates: 2 cx + 1 ccx
        multi = [g for g in results["gate_list"] if g.get("arity", 0) >= 2]
        self.assertEqual(len(multi), 3)


# ---------------------------------------------------------------------------
# Step 7: Output generation (toQasm / toQasmFF with spec)
# ---------------------------------------------------------------------------

class TestStep7Output(unittest.TestCase):
    """Step 7: Output QASM from solution; verify gate count and SubgraphMatches."""

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_output_2q_only(self):
        """After solving 2q-only circuit, toQasmFF output has same 2q count and every 2q on an edge."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_2q_only.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        allowed_2 = set(tuple(t) for t in hw.get_subgraph_matches(spec, 2, None))
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "prob")
            sname = os.path.join(td, "sol")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=30)
            qasm_str = sc.toQasmFF(
                qasm_dest, cm, 1, 1, sname,
                gate_list=results["gate_list"], chunk_ranges=results["chunk_ranges"],
                spec=spec, match_counts_per_chunk=results["match_counts_per_chunk"]
            )
        # Parse output: count 2q gates and check each (p1,p2) in edges
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm_str)
        two_q = 0
        for j in range(len(circ)):
            qs = circ[j][1]
            if len(qs) >= 2:
                two_q += 1
                phys = tuple(circ.find_bit(q)[0] for q in qs)
                self.assertIn(phys, allowed_2, f"2q gate qubits {phys} not in SubgraphMatches(2)")
        self.assertEqual(two_q, 2)

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_output_1q_2q_order(self):
        """Circuit H–CX–X–CX: output order H, CX, X, CX and single-qubit use correct mapping."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_1q_2q.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "prob")
            sname = os.path.join(td, "sol")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=30)
            qasm_str = sc.toQasmFF(
                qasm_dest, cm, 1, 1, sname,
                gate_list=results["gate_list"], chunk_ranges=results["chunk_ranges"],
                spec=spec, match_counts_per_chunk=results["match_counts_per_chunk"]
            )
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm_str)
        names = [circ[j][0].name for j in range(len(circ))]
        self.assertEqual(names, ["h", "cx", "x", "cx"], "gate order")

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_output_3q_mapping(self):
        """After solving circuit with 3q gate, output 3q gate's physical qubits in SubgraphMatches for 3."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_2q_3q.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        allowed_3 = set(tuple(t) for t in hw.get_subgraph_matches(spec, 3, None))
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "prob")
            sname = os.path.join(td, "sol")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=45)
            qasm_str = sc.toQasmFF(
                qasm_dest, cm, 1, 1, sname,
                gate_list=results["gate_list"], chunk_ranges=results["chunk_ranges"],
                spec=spec, match_counts_per_chunk=results["match_counts_per_chunk"]
            )
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm_str)
        for j in range(len(circ)):
            qs = circ[j][1]
            if len(qs) == 3:
                phys = tuple(circ.find_bit(q)[0] for q in qs)
                self.assertIn(phys, allowed_3, f"3q gate {phys} not in SubgraphMatches")
                break
        else:
            self.fail("No 3-qubit gate in output")

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_swaps_inserted_before_correct_gate(self):
        """Every SWAP in output is part of a block immediately before a multi-qubit gate."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_1q_2q.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "p")
            sname = os.path.join(td, "s")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=30)
            qasm_str = sc.toQasmFF(
                qasm_dest, cm, 1, 1, sname,
                gate_list=results["gate_list"], chunk_ranges=results["chunk_ranges"],
                spec=spec, match_counts_per_chunk=results["match_counts_per_chunk"]
            )
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm_str)
        # Structural check: any run of swaps must be immediately followed by a multi-qubit gate
        i = 0
        while i < len(circ):
            if circ[i][0].name == "swap":
                # Next non-swap must be multi-qubit
                j = i
                while j < len(circ) and circ[j][0].name == "swap":
                    j += 1
                if j < len(circ):
                    self.assertGreaterEqual(len(circ[j][1]), 2, "SWAPs must be immediately before a multi-qubit gate")
                i = j
            else:
                i += 1


# ---------------------------------------------------------------------------
# Step 8: E2E and flow tests
# ---------------------------------------------------------------------------

class TestStep8E2E(unittest.TestCase):
    """Step 8: End-to-end with hardware_spec; missing subgraph error."""

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_e2e_2q_only(self):
        """Full run: 2q circuit + JSON with edges; output valid and all 2q on edges."""
        import satmap
        prog = str(FIXTURES_DIR / "circuit_2q_only.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        cm = hw.build_cm_from_spec(hw.load_spec(spec_path))
        stats, qasm = satmap.transpile(prog, cm, 1, "aux_files/prob_e2e", "aux_files/sol_e2e", slice_size=5, max_sat_time=30, hardware_spec=spec_path)
        self.assertIsNotNone(stats)
        self.assertIsNotNone(qasm)
        allowed = {tuple(e) for e in hw.get_edges(hw.load_spec(spec_path))}
        allowed |= {(b, a) for (a, b) in allowed}
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm)
        for j in range(len(circ)):
            qs = circ[j][1]
            if len(qs) >= 2:
                phys = tuple(circ.find_bit(q)[0] for q in qs)
                self.assertIn(phys, allowed)

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_e2e_3q(self):
        """Full run: circuit with 3q gate + JSON with subgraph_matches for 3."""
        import satmap
        prog = str(FIXTURES_DIR / "circuit_2q_3q.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        stats, qasm = satmap.transpile(prog, cm, 1, "aux_files/prob_e2e3", "aux_files/sol_e2e3", slice_size=5, max_sat_time=45, hardware_spec=spec_path)
        self.assertIsNotNone(stats)
        self.assertIsNotNone(qasm)
        allowed_3 = set(tuple(t) for t in hw.get_subgraph_matches(spec, 3, None))
        circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm)
        for j in range(len(circ)):
            if len(circ[j][1]) == 3:
                phys = tuple(circ.find_bit(q)[0] for q in circ[j][1])
                self.assertIn(phys, allowed_3)
                break

    def test_e2e_missing_subgraph(self):
        """Circuit with 3q gate, JSON without subgraph_matches['3']; clear error (no solver needed)."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_2q_3q.qasm")
        spec = hw.load_spec({"num_physical_qubits": 4, "edges": [[0, 1], [1, 2], [2, 3]], "subgraph_matches": {"2": {"edge": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]}}})
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "p")
            sname = os.path.join(td, "s")
            import shutil
            qasm_dest = os.path.join(td, "qiskit-circuit_2q_3q.qasm")
            shutil.copy(prog, qasm_dest)
            with self.assertRaises(ValueError) as ctx:
                sc.solve(qasm_dest, cm, swapNum=1, chunks=1, pname=pname, sname=sname, spec=spec, time_wbo_max=5)
            self.assertIn("subgraph_matches", str(ctx.exception).lower())

    @unittest.skipUnless(_open_wbo_available(), "Open-WBO-Inc not found")
    def test_e2e_chunking(self):
        """Many multi-qubit gates, 2 chunks; two solution files and complete output. Verifies chunk-boundary consistency."""
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_many_2q_for_chunking.qasm")
        spec_path = str(FIXTURES_DIR / "hw_spec_rich.json")
        spec = hw.load_spec(spec_path)
        cm = hw.build_cm_from_spec(spec)
        with tempfile.TemporaryDirectory() as td:
            pname = os.path.join(td, "p")
            sname = os.path.join(td, "s")
            import shutil
            tail = os.path.basename(prog)
            qasm_dest = os.path.join(td, "qiskit-" + tail)
            shutil.copy(prog, qasm_dest)
            results = sc.solve(qasm_dest, cm, swapNum=1, chunks=2, pname=pname, sname=sname, spec=spec, time_wbo_max=60)
            self.assertEqual(results["chunks"], 2)
            self.assertEqual(len(results["chunk_ranges"]), 2)
            self.assertEqual(len(results["match_counts_per_chunk"]), 2)
            multi = [g for g in results["gate_list"] if g.get("arity", 0) >= 2]
            self.assertEqual(len(multi), 6)
            for i in range(2):
                sol_src = os.path.join(td, "s-chnk" + str(i) + ".txt")
                self.assertTrue(os.path.isfile(sol_src), "Missing solution file " + sol_src)
            # Chunk-boundary consistency: last-step mapping of chunk 0 == step-0 mapping of chunk 1
            from common import extract_qubits
            log_num = max(extract_qubits(results["gate_list"])) + 1
            phys_num = len(cm)
            prev_size = results["chunk_ranges"][0][2]
            map0_all = list(sc.mappingVars(
                sc.readMaxSatOutput, phys_num, log_num, prev_size, 1,
                os.path.join(td, "s-chnk0.txt"), match_counts=results["match_counts_per_chunk"][0]
            ))
            map0_last = [(p, l) for (p, l, k) in map0_all if k == prev_size - 1]
            size1 = results["chunk_ranges"][1][2]
            map1_all = list(sc.mappingVars(
                sc.readMaxSatOutput, phys_num, log_num, size1, 1,
                os.path.join(td, "s-chnk1.txt"), match_counts=results["match_counts_per_chunk"][1]
            ))
            map1_0 = [(p, l) for (p, l, k) in map1_all if k == 0]
            self.assertEqual(set(map0_last), set(map1_0), "Chunk boundary: last step of chunk0 should equal step0 of chunk1")
            qasm_str = sc.toQasmFF(
                qasm_dest, cm, 1, 2, sname,
                gate_list=results["gate_list"], chunk_ranges=results["chunk_ranges"],
                spec=spec, match_counts_per_chunk=results["match_counts_per_chunk"]
            )
            circ = __import__("qiskit").QuantumCircuit.from_qasm_str(qasm_str)
            # Original circuit has 6 multi-qubit gates; output may add SWAPs, so total >= 6
            two_q_count = sum(1 for j in range(len(circ)) if len(circ[j][1]) >= 2)
            self.assertGreaterEqual(two_q_count, 6)

# ---------------------------------------------------------------------------
# Flow tests (Step 1→2, 2→3, ...)
# ---------------------------------------------------------------------------

class TestFlow(unittest.TestCase):
    """Flow: gate list + spec → chunk ranges → constraint count; rich spec and multi-arity circuit."""

    def test_flow_gate_list_chunk_ranges_spec(self):
        """Gate list from circuit; chunk ranges; spec has 2/3/4/5 types."""
        from common import extract_gates, get_chunk_ranges, get_multi_qubit_indices
        prog = str(FIXTURES_DIR / "circuit_multi_arity.qasm")
        spec = hw.load_spec(str(FIXTURES_DIR / "hw_spec_rich.json"))
        gate_list = extract_gates(prog)
        self.assertGreaterEqual(len(gate_list), 5)
        multi = [g for g in gate_list if g.get("arity", 0) >= 2]
        self.assertEqual(len(multi), 5)  # cx, ccx, custom4, custom5, cx
        indices = get_multi_qubit_indices(gate_list)
        self.assertEqual(len(indices), 5)
        ranges = get_chunk_ranges(gate_list, 2)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][2] + ranges[1][2], 5)

    def test_flow_rich_spec_subgraph_types(self):
        """Rich spec: get_subgraph_matches for 2,3,4,5 and multiple types (triangle, line3, line4, square, line5)."""
        spec = hw.load_spec(str(FIXTURES_DIR / "hw_spec_rich.json"))
        self.assertEqual(hw.get_num_physical_qubits(spec), 6)
        self.assertGreater(len(hw.get_edges(spec)), 0)
        for arity in (2, 3, 4, 5):
            types = hw.get_subgraph_match_types(spec, arity)
            self.assertGreater(len(types), 0, f"arity {arity} has no types")
            union = hw.get_subgraph_matches(spec, arity, None)
            self.assertGreater(len(union), 0)
        self.assertIn("triangle", hw.get_subgraph_match_types(spec, 3))
        self.assertIn("line3", hw.get_subgraph_match_types(spec, 3))
        self.assertIn("line4", hw.get_subgraph_match_types(spec, 4))
        self.assertIn("square", hw.get_subgraph_match_types(spec, 4))
        self.assertIn("line5", hw.get_subgraph_match_types(spec, 5))

    def test_flow_multi_arity_constraint_generation(self):
        """Circuit with 1,2,3,4,5 qubit gates + rich spec: constraint generation succeeds (no solve)."""
        from common import extract_gates, get_chunk_ranges, extract_qubits
        import satmap_core as sc
        prog = str(FIXTURES_DIR / "circuit_multi_arity.qasm")
        spec = hw.load_spec(str(FIXTURES_DIR / "hw_spec_rich.json"))
        cm = hw.build_cm_from_spec(spec)
        gate_list = extract_gates(prog)
        log_num = max(extract_qubits(gate_list)) + 1
        multi = [g for g in gate_list if g.get("arity", 0) >= 2]
        self.assertEqual(len(multi), 5)  # cx, ccx, custom4, custom5, cx
        ranges = get_chunk_ranges(gate_list, 1)
        self.assertEqual(len(ranges), 1)
        start, end_excl, currentSize = ranges[0]
        multi_chunk = [g for g in gate_list[start:end_excl] if g.get("arity", 0) >= 2]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wcnf", delete=False) as tmp:
            path = tmp.name
        try:
            s, match_counts = sc.generateAndWriteClauses(
                log_num, [], [], cm, 1, [], path,
                routing=True, spec=spec, multi_qubit_gates=multi_chunk
            )
            self.assertIsNotNone(match_counts)
            self.assertEqual(len(match_counts), 5)
            # Arity 2 has many matches; 3 has several; 4 and 5 have fewer
            self.assertGreater(match_counts[0], 0)
            self.assertGreater(match_counts[1], 0)
            self.assertGreater(match_counts[2], 0)
            self.assertGreater(match_counts[3], 0)
            self.assertGreater(match_counts[4], 0)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
