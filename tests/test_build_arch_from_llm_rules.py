"""
Tests for scripts/build_arch_from_llm_rules.py (subgraph matcher).

Run from project root with:
  PYTHONPATH=src python -m unittest tests.test_build_arch_from_llm_rules -v

Requires: pip install networkx
"""

import json
import unittest
from pathlib import Path

import sys
_here = Path(__file__).resolve().parent
_root = _here.parent
_src = _root / "src"
_scripts = _root / "scripts"
for p in (_src, _scripts):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from build_arch_from_llm_rules import (
        load_raw_hardware,
        load_llm_rules,
        build_target_spec,
        derive_arity2_edge_tuples,
        type_name_for_rule,
    )
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    IMPORT_ERR = e

import hardware_spec as hw


def _skip_if_no_networkx():
    if not HAS_DEPS:
        raise unittest.SkipTest(f"build_arch_from_llm_rules or deps not available: {IMPORT_ERR}")


class TestBuildArchFromLlmRules(unittest.TestCase):
    def test_arity2_derivation(self):
        _skip_if_no_networkx()
        edges = [[0, 1], [1, 2]]
        out = derive_arity2_edge_tuples(edges)
        self.assertEqual(len(out), 4)
        self.assertIn([0, 1], out)
        self.assertIn([1, 0], out)
        self.assertIn([1, 2], out)
        self.assertIn([2, 1], out)

    def test_type_name_for_rule(self):
        _skip_if_no_networkx()
        self.assertEqual(type_name_for_rule({"nQubits": 3, "shape": "line"}), "line3")
        self.assertEqual(type_name_for_rule({"nQubits": 4, "shape": "line"}), "line4")
        self.assertEqual(type_name_for_rule({"nQubits": 3, "shape": "triangle"}), "triangle")

    def test_build_target_spec_smoke(self):
        _skip_if_no_networkx()
        raw = {"num_physical_qubits": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]}
        llm_rules = [
            {"nQubits": 3, "shape": "line", "edges": [[0, 1], [1, 2]]},
        ]
        spec = build_target_spec(raw, llm_rules)
        self.assertIn("subgraph_matches", spec)
        self.assertEqual(spec["num_physical_qubits"], 5)
        self.assertEqual(spec["subgraph_matches"]["2"]["edge"], derive_arity2_edge_tuples(raw["edges"]))
        line3 = spec["subgraph_matches"]["3"]["line3"]
        # Path 0-1-2: at least (0,1,2), (1,2,3), (2,3,4); reverse order gives 6 total
        self.assertIn([0, 1, 2], line3)
        self.assertIn([1, 2, 3], line3)
        self.assertIn([2, 3, 4], line3)
        self.assertGreaterEqual(len(line3), 3)

    def test_build_target_spec_loadable_by_hardware_spec(self):
        _skip_if_no_networkx()
        raw = {"num_physical_qubits": 4, "edges": [[0, 1], [1, 2], [2, 3], [0, 2]]}
        llm_rules = [
            {"nQubits": 3, "shape": "line", "edges": [[0, 1], [1, 2]]},
            {"nQubits": 3, "shape": "triangle", "edges": [[0, 1], [1, 2], [2, 0]]},
        ]
        spec = build_target_spec(raw, llm_rules)
        loaded = hw.load_spec(spec)
        self.assertEqual(hw.get_num_physical_qubits(loaded), 4)
        self.assertEqual(len(hw.get_subgraph_matches(loaded, 2)), 8)  # 4 edges * 2
        line3 = hw.get_subgraph_matches(loaded, 3, "line3")
        triangle = hw.get_subgraph_matches(loaded, 3, "triangle")
        self.assertGreater(len(line3), 0)
        self.assertGreater(len(triangle), 0)

    def test_tokyo_edges_and_llm_example(self):
        _skip_if_no_networkx()
        raw_path = _root / "examples-multi" / "tokyo-edges.json"
        llm_path = _root / "scripts" / "llm_rules_example.json"
        if not raw_path.exists() or not llm_path.exists():
            self.skipTest("examples-multi/tokyo-edges.json or scripts/llm_rules_example.json not found")
        raw = load_raw_hardware(raw_path)
        llm_rules = load_llm_rules(llm_path)
        spec = build_target_spec(raw, llm_rules)
        loaded = hw.load_spec(spec)
        self.assertEqual(hw.get_num_physical_qubits(loaded), 20)
        self.assertEqual(len(hw.get_edges(loaded)), 43)
        self.assertIn("2", spec["subgraph_matches"])
        self.assertIn("edge", spec["subgraph_matches"]["2"])
        self.assertEqual(len(spec["subgraph_matches"]["2"]["edge"]), 86)
        self.assertIn("3", spec["subgraph_matches"])
        self.assertIn("line3", spec["subgraph_matches"]["3"])
        self.assertIn("4", spec["subgraph_matches"])
        self.assertIn("line4", spec["subgraph_matches"]["4"])
        self.assertGreater(len(spec["subgraph_matches"]["3"]["line3"]), 0)
        self.assertGreater(len(spec["subgraph_matches"]["4"]["line4"]), 0)


if __name__ == "__main__":
    unittest.main()
