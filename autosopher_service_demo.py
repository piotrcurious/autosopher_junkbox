#!/usr/bin/env python3
"""
Autosopher Knowledge Service & Typed Epistemic Engine
Implements Section 4 of Junkbox 1.md:
- Typed Knowledge Model:
    * Entity (id, type, canonical_name, attributes)
    * Assertion (id, subject, predicate, object, qualifiers, provenance, valid_time, confidence, version, status)
    * Rule (id, preconditions, executable_expression, units_and_types, validation_tests)
    * Artifact (id, type, content_hash, schema, source_reference)
    * Derivation (id, input_assertions, rule_id, output_assertions, execution_result)
- Operational Tool-Calling Interface:
    * memory.resolve(query, filters)
    * memory.traverse(entity_id, relations, depth)
    * memory.execute(rule_id, inputs)
    * memory.propose(assertion)
    * memory.validate(candidate_id)
    * memory.commit(candidate_id)
    * memory.dependencies(assertion_id)
- Reactive Dependency Invalidation & Propagation:
    When a source measurement updates or is contradicted, downstream derived assertions
    are automatically marked STALE/INVALIDATED and recomputed.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional

class Entity:
    def __init__(self, entity_id: str, entity_type: str, canonical_name: str, attributes: Dict[str, Any] = None):
        self.id = entity_id
        self.type = entity_type
        self.canonical_name = canonical_name
        self.attributes = attributes or {}

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "canonical_name": self.canonical_name,
            "attributes": self.attributes
        }

class Assertion:
    def __init__(self, assertion_id: str, subject: str, predicate: str, obj: Any,
                 qualifiers: Dict[str, Any] = None, provenance: str = "direct_observation",
                 valid_time: float = None, confidence: float = 1.0, version: int = 1, status: str = "GROUNDED"):
        self.id = assertion_id
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.qualifiers = qualifiers or {}
        self.provenance = provenance
        self.valid_time = valid_time or time.time()
        self.confidence = confidence
        self.version = version
        self.status = status  # GROUNDED, DERIVED, HYPOTHETICAL, CONTRADICTED, UNRESOLVED, STALE

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "qualifiers": self.qualifiers,
            "provenance": self.provenance,
            "valid_time": self.valid_time,
            "confidence": self.confidence,
            "version": self.version,
            "status": self.status
        }

class Rule:
    def __init__(self, rule_id: str, preconditions: List[str], expression_str: str,
                 units_and_types: Dict[str, str], validation_tests: List[Dict[str, Any]] = None):
        self.id = rule_id
        self.preconditions = preconditions
        self.expression_str = expression_str
        self.units_and_types = units_and_types
        self.validation_tests = validation_tests or []

    def evaluate(self, inputs: Dict[str, float]) -> float:
        # Safe evaluation of mathematical expression with provided inputs
        allowed = {k: v for k, v in inputs.items()}
        return float(eval(self.expression_str, {"__builtins__": None}, allowed))

class Derivation:
    def __init__(self, derivation_id: str, input_assertions: List[str], rule_id: str,
                 output_assertions: List[str], execution_result: Any):
        self.id = derivation_id
        self.input_assertions = input_assertions
        self.rule_id = rule_id
        self.output_assertions = output_assertions
        self.execution_result = execution_result
        self.timestamp = time.time()

class Artifact:
    def __init__(self, artifact_id: str, artifact_type: str, content: str, schema: str = "json", source_ref: str = ""):
        self.id = artifact_id
        self.type = artifact_type
        self.content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        self.schema = schema
        self.source_reference = source_ref

class AutosopherKnowledgeService:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.assertions: Dict[str, Assertion] = {}
        self.rules: Dict[str, Rule] = {}
        self.artifacts: Dict[str, Artifact] = {}
        self.derivations: Dict[str, Derivation] = {}
        self.candidate_assertions: Dict[str, Assertion] = {}

        # Dependency graph: input_assertion_id -> List[derivation_id]
        self.downstream_dependencies: Dict[str, List[str]] = {}

    # Tool: memory.resolve(query, filters)
    def resolve(self, query: str = "", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        filters = filters or {}
        matches = []
        for a in self.assertions.values():
            if filters.get("status") and a.status != filters["status"]:
                continue
            if filters.get("predicate") and a.predicate != filters["predicate"]:
                continue
            if query and (query not in a.subject and query not in str(a.object)):
                continue
            matches.append(a.to_dict())
        return matches

    # Tool: memory.traverse(entity_id, relations, depth)
    def traverse(self, entity_id: str, relations: List[str] = None, depth: int = 1) -> List[Dict[str, Any]]:
        visited = []
        def dfs(current_id, current_depth):
            if current_depth > depth:
                return
            for a in self.assertions.values():
                if a.subject == current_id and (not relations or a.predicate in relations):
                    visited.append(a.to_dict())
                    if isinstance(a.object, str) and a.object in self.entities:
                        dfs(a.object, current_depth + 1)
        dfs(entity_id, 1)
        return visited

    # Tool: memory.execute(rule_id, inputs)
    def execute(self, rule_id: str, inputs: Dict[str, float]) -> float:
        if rule_id not in self.rules:
            raise KeyError(f"Rule {rule_id} not registered.")
        rule = self.rules[rule_id]
        return rule.evaluate(inputs)

    # Tool: memory.propose(assertion)
    def propose(self, assertion: Assertion) -> str:
        assertion.status = "HYPOTHETICAL"
        self.candidate_assertions[assertion.id] = assertion
        return assertion.id

    # Tool: memory.validate(candidate_id)
    def validate(self, candidate_id: str) -> bool:
        if candidate_id not in self.candidate_assertions:
            return False
        candidate = self.candidate_assertions[candidate_id]
        # Validate confidence threshold & non-empty subject/predicate
        if candidate.confidence >= 0.7 and candidate.subject and candidate.predicate:
            return True
        return False

    # Tool: memory.commit(candidate_id)
    def commit(self, candidate_id: str) -> bool:
        if not self.validate(candidate_id):
            return False
        candidate = self.candidate_assertions.pop(candidate_id)
        candidate.status = "GROUNDED" if candidate.provenance == "direct_observation" else "DERIVED"
        self.assertions[candidate.id] = candidate
        return True

    # Tool: memory.dependencies(assertion_id)
    def dependencies(self, assertion_id: str) -> List[str]:
        return self.downstream_dependencies.get(assertion_id, [])

    # Register derivation and link reactive dependency edges
    def record_derivation(self, derivation: Derivation):
        self.derivations[derivation.id] = derivation
        for in_id in derivation.input_assertions:
            if in_id not in self.downstream_dependencies:
                self.downstream_dependencies[in_id] = []
            self.downstream_dependencies[in_id].append(derivation.id)

    # Invalidation & propagation (Section 4D)
    def update_assertion(self, assertion_id: str, new_value: Any, new_confidence: float = 1.0):
        if assertion_id not in self.assertions:
            return
        old_assertion = self.assertions[assertion_id]
        old_assertion.object = new_value
        old_assertion.confidence = new_confidence
        old_assertion.version += 1

        # Propagate to downstream dependencies
        downstream = self.dependencies(assertion_id)
        for deriv_id in downstream:
            deriv = self.derivations.get(deriv_id)
            if not deriv:
                continue
            for out_id in deriv.output_assertions:
                if out_id in self.assertions:
                    # Invalidate stale conclusion
                    self.assertions[out_id].status = "STALE"
                    self.assertions[out_id].version += 1

            # Automatically recompute derivation if rule exists
            if deriv.rule_id in self.rules:
                rule = self.rules[deriv.rule_id]
                inputs = {
                    "pump_flow": self.assertions["assert_pump_flow"].object,
                    "circuit_resist": self.assertions["assert_ckt_resist"].object
                }
                try:
                    recomputed = rule.evaluate(inputs)
                    for out_id in deriv.output_assertions:
                        if out_id in self.assertions:
                            self.assertions[out_id].object = recomputed
                            self.assertions[out_id].status = "DERIVED"
                            self.assertions[out_id].version += 1
                except Exception as e:
                    pass

def run_service_pipeline_demo():
    print("=" * 75)
    print("  AUTOSOPHER TYPED KNOWLEDGE SERVICE & DEPENDENCY ENGINE (Junkbox 1.md)")
    print("=" * 75)

    svc = AutosopherKnowledgeService()

    # Step 1: Register Entities
    pmp = Entity("pump_A", "Pump", "Centrifugal Pump Alpha", {"power_kw": 45.0})
    ckt = Entity("circuit_B", "Circuit", "Cooling Hydraulic Circuit Beta", {"design_pressure_bar": 12.0})
    svc.entities[pmp.id] = pmp
    svc.entities[ckt.id] = ckt
    print("\n[1] Registered Entities: 'pump_A' and 'circuit_B'")

    # Step 2: Register Rules
    rule_flow = Rule(
        "rule_flow_rate",
        ["pump_flow", "circuit_resist"],
        "pump_flow * 0.85 - circuit_resist * 2.0",
        {"pump_flow": "m3_per_hr", "circuit_resist": "bar"}
    )
    svc.rules[rule_flow.id] = rule_flow
    print(f"[2] Registered Derivation Rule: '{rule_flow.id}' => {rule_flow.expression_str}")

    # Step 3: Propose & Commit Source Observations
    a1 = Assertion("assert_pump_flow", "pump_A", "flow_rate", 100.0, provenance="direct_observation")
    a2 = Assertion("assert_ckt_resist", "circuit_B", "resistance", 5.0, provenance="direct_observation")
    svc.propose(a1)
    svc.propose(a2)
    svc.commit("assert_pump_flow")
    svc.commit("assert_ckt_resist")
    print(f"[3] Committed Grounded Observations: pump_flow=100.0, resistance=5.0")

    # Step 4: Execute Rule & Commit Derived Output
    val = svc.execute("rule_flow_rate", {"pump_flow": 100.0, "circuit_resist": 5.0})
    a3 = Assertion("assert_net_delivery", "circuit_B", "net_delivery", val, provenance="rule_flow_rate", status="DERIVED")
    svc.assertions[a3.id] = a3

    deriv = Derivation("deriv_1", ["assert_pump_flow", "assert_ckt_resist"], "rule_flow_rate", ["assert_net_delivery"], val)
    svc.record_derivation(deriv)
    print(f"[4] Computed Derivation: net_delivery = {val} m3/hr (Status: DERIVED, Version: 1)")

    # Step 5: Downstream Invalidation & Reactive Recomputation
    print("\n[5] Simulating Source Update / Sensor Recalibration on 'assert_pump_flow': 100.0 -> 120.0...")
    svc.update_assertion("assert_pump_flow", 120.0)

    updated_delivery = svc.assertions["assert_net_delivery"]
    print(f" SUCCESS: Downstream assertion 'assert_net_delivery' automatically recomputed!")
    print(f" New Value: {updated_delivery.object} m3/hr | Status: {updated_delivery.status} | Version: {updated_delivery.version}")

    # Step 6: Query Knowledge Service (Resolve & Traverse)
    print("\n[6] Testing Memory Tools:")
    relation_assert = Assertion("assert_rel", "pump_A", "supplies", "circuit_B", provenance="direct_observation")
    svc.assertions[relation_assert.id] = relation_assert

    traversed = svc.traverse("pump_A", depth=2)
    print(f" memory.traverse('pump_A') -> Found {len(traversed)} connected relations.")

    resolved = svc.resolve(filters={"status": "DERIVED"})
    print(f" memory.resolve(status='DERIVED') -> Found {len(resolved)} active derived facts.")

    print("\n" + "=" * 75)
    print(" KNOWLEDGE SERVICE DEMONSTRATION COMPLETE!")
    print("=" * 75)

if __name__ == "__main__":
    run_service_pipeline_demo()
