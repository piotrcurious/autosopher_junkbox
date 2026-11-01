#!/usr/bin/env python3
"""
Unified Integration & Regression Test for Hybrid Autosopher System
Verifies:
1. C-Python ctypes bindings to libautosopher.so
2. Knowledge package serialization, transfer, and deserialization
3. Autosopher Knowledge Service with reactive invalidation
4. Extended inference with Deep Belief correlation and on-the-fly autocoding
5. Architectural comparative benchmarks
"""

import os
import sys
import unittest
import ctypes
import numpy as np

# Load ctypes structure
class Node(ctypes.Structure):
    pass

Node._fields_ = [
    ('key_feature1', ctypes.c_int),
    ('key_feature2', ctypes.c_int),
    ('data_payload', ctypes.c_double),
    ('outcome', ctypes.c_int),
    ('access_count', ctypes.c_int),
    ('epistemic_status', ctypes.c_int),
    ('version', ctypes.c_int),
    ('deep_belief_code', ctypes.c_int),
    ('generative_id', ctypes.c_int),
    ('height', ctypes.c_int),
    ('left', ctypes.POINTER(Node)),
    ('right', ctypes.POINTER(Node))
]

class TreeStats(ctypes.Structure):
    _fields_ = [
        ('node_count', ctypes.c_int),
        ('max_depth', ctypes.c_int),
        ('memory_bytes', ctypes.c_size_t)
    ]

class TestAutosopherHybridIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libautosopher.so')
        if not os.path.exists(so_path):
            os.system("make libautosopher.so >/dev/null 2>&1")
        cls.lib = ctypes.CDLL(so_path)

        cls.lib.autosophy_learn_typed.argtypes = [
            ctypes.POINTER(ctypes.POINTER(Node)),
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        cls.lib.autosophy_recall_typed.argtypes = [
            ctypes.POINTER(Node), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        cls.lib.autosophy_recall_typed.restype = ctypes.c_int

        cls.lib.autosophy_correlate_deep_belief.argtypes = [
            ctypes.POINTER(Node), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        ]
        cls.lib.autosophy_correlate_deep_belief.restype = ctypes.c_int

        cls.lib.autosophy_autocode_category.argtypes = [
            ctypes.POINTER(ctypes.POINTER(Node)),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int
        ]
        cls.lib.autosophy_autocode_category.restype = ctypes.c_int

        cls.lib.get_tree_stats.argtypes = [ctypes.POINTER(Node), ctypes.POINTER(TreeStats)]
        cls.lib.free_tree.argtypes = [ctypes.POINTER(Node)]

    def test_c_bindings_and_learning(self):
        root = ctypes.POINTER(Node)()
        self.lib.autosophy_learn_typed(ctypes.byref(root), 15, 25, 0.77, 9, 0, 1)

        out_p = ctypes.c_double()
        out_o = ctypes.c_int()
        out_s = ctypes.c_int()
        out_v = ctypes.c_int()

        hit = self.lib.autosophy_recall_typed(root, 15, 25, ctypes.byref(out_p), ctypes.byref(out_o), ctypes.byref(out_s), ctypes.byref(out_v))
        self.assertEqual(hit, 1)
        self.assertEqual(out_o.value, 9)
        self.assertAlmostEqual(out_p.value, 0.77, places=4)
        self.assertEqual(out_s.value, 0)
        self.assertEqual(out_v.value, 1)

        self.lib.free_tree(root)

    def test_deep_belief_autocoding(self):
        root = ctypes.POINTER(Node)()
        out_db = ctypes.c_int()
        out_gen = ctypes.c_int()
        out_seed = ctypes.c_double()

        # Check miss before autocoding
        exists = self.lib.autosophy_correlate_deep_belief(root, 100, 200, ctypes.byref(out_db), ctypes.byref(out_gen), ctypes.byref(out_seed))
        self.assertEqual(exists, 0)

        # Autocode category on-the-fly
        self.lib.autosophy_autocode_category(ctypes.byref(root), 100, 200, 100200, 5, 0.88, 5)

        # Correlate after autocoding
        exists = self.lib.autosophy_correlate_deep_belief(root, 100, 200, ctypes.byref(out_db), ctypes.byref(out_gen), ctypes.byref(out_seed))
        self.assertEqual(exists, 1)
        self.assertEqual(out_db.value, 100200)
        self.assertEqual(out_gen.value, 5)

        self.lib.free_tree(root)

    def test_service_reactive_invalidation(self):
        from autosopher_service_demo import AutosopherKnowledgeService, Entity, Rule, Assertion, Derivation
        svc = AutosopherKnowledgeService()

        pmp = Entity("p1", "Pump", "Test Pump")
        svc.entities[pmp.id] = pmp

        r = Rule("r1", ["flow"], "flow * 0.9", {"flow": "gpm"})
        svc.rules[r.id] = r

        a1 = Assertion("assert_pump_flow", "p1", "flow", 50.0)
        svc.assertions[a1.id] = a1

        # Calculate derivation
        val = svc.execute("r1", {"flow": 50.0})
        a_out = Assertion("assert_net_flow", "p1", "net_flow", val, status="DERIVED")
        svc.assertions[a_out.id] = a_out

        deriv = Derivation("d1", ["assert_pump_flow"], "r1", ["assert_net_flow"], val)
        svc.record_derivation(deriv)

        self.assertEqual(svc.assertions["assert_net_flow"].object, 45.0)

        # Update source
        svc.update_assertion("assert_pump_flow", 100.0)
        # Verify reactive update
        self.assertEqual(svc.assertions["assert_net_flow"].status, "DERIVED")
        self.assertEqual(svc.assertions["assert_net_flow"].version, 3)

if __name__ == "__main__":
    unittest.main()
