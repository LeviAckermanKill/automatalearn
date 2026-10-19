"""Regular Positive and Negative Inference (RPNI) algorithm.

Passive grammatical inference algorithm learning minimal deterministic finite
automata from positive and negative string samples via state merging and
deterministic folding with consistency verification.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from automatalearn.core.dfa import DFA, State
from automatalearn.core.minimize import minimize_dfa
from automatalearn.learning.prefix_tree import PrefixTreeAcceptor


@dataclass
class RPNITraceEntry:
    """Detailed log record of an RPNI step."""
    step_num: int
    blue_state: State
    red_state: Optional[State]
    action: str  # "MERGE_SUCCESS", "MERGE_REJECTED", "PROMOTED"
    reason: str
    num_states_after: int


@dataclass
class RPNIResult:
    """Output container for RPNI learning."""
    learned_dfa: DFA
    minimized_dfa: DFA
    pta_state_count: int
    final_state_count: int
    merges_attempted: int
    merges_accepted: int
    execution_time_ms: float
    trace_log: List[RPNITraceEntry] = field(default_factory=list)


class RPNILearner:
    """Learns a DFA from positive and negative string samples using RPNI."""

    def __init__(self, alphabet: List[str]) -> None:
        self.alphabet = sorted(list(alphabet))

    def _fold(
        self,
        current_dfa: DFA,
        q_r: State,
        q_b: State,
    ) -> Optional[DFA]:
        """Merge state q_b into q_r and fold non-deterministic branchings via disjoint-set."""
        parent: Dict[State, State] = {s: s for s in current_dfa.states}

        def find(s: State) -> State:
            if parent[s] != s:
                parent[s] = find(parent[s])
            return parent[s]

        def union(s1: State, s2: State) -> None:
            r1, r2 = find(s1), find(s2)
            if r1 != r2:
                # Prioritize keeping lower id / q_r representative
                parent[r2] = r1

        # Initial merge
        union(q_r, q_b)
        fold_queue: List[Tuple[State, State]] = [(q_r, q_b)]

        while fold_queue:
            u, v = fold_queue.pop(0)
            rep_u, rep_v = find(u), find(v)

            # Check outgoing transitions on each alphabet symbol
            for sym in self.alphabet:
                targets: Set[State] = set()
                # Check transitions from all states in the partition
                for (st, a), dst in current_dfa.transitions.items():
                    if a == sym and find(st) in (rep_u, rep_v):
                        targets.add(find(dst))

                if len(targets) > 1:
                    t_list = list(targets)
                    base = t_list[0]
                    for other in t_list[1:]:
                        if find(base) != find(other):
                            union(base, other)
                            fold_queue.append((base, other))

        # Build candidate quotient automaton
        quotient_states: Set[State] = {find(s) for s in current_dfa.states}
        quotient_start: State = find(current_dfa.start_state)
        quotient_accept: Set[State] = {find(s) for s in current_dfa.accept_states}
        quotient_transitions: Dict[Tuple[State, str], State] = {}

        for (st, sym), dst in current_dfa.transitions.items():
            rep_st = find(st)
            rep_dst = find(dst)
            if (rep_st, sym) in quotient_transitions:
                if quotient_transitions[(rep_st, sym)] != rep_dst:
                    # Still non-deterministic (should not happen after complete folding)
                    return None
            else:
                quotient_transitions[(rep_st, sym)] = rep_dst

        return DFA(
            states=quotient_states,
            alphabet=set(self.alphabet),
            transitions=quotient_transitions,
            start_state=quotient_start,
            accept_states=quotient_accept,
            name="RPNI_Candidate",
        )

    def _is_consistent(self, candidate_dfa: DFA, negative_samples: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if candidate DFA rejects all negative training samples."""
        for neg in negative_samples:
            if candidate_dfa.accepts(neg):
                return False, neg
        return True, None

    def learn(
        self,
        positive_samples: List[str],
        negative_samples: List[str],
    ) -> RPNIResult:
        """Execute RPNI state merging to learn a DFA.

        Args:
            positive_samples: List of strings in the target language.
            negative_samples: List of strings not in the target language.

        Returns:
            RPNIResult containing learned DFA, minimized DFA, metrics, and trace log.
        """
        import time
        t0 = time.perf_counter()

        # Step 1: Build Prefix Tree Acceptor (PTA) from positive samples
        pta_builder = PrefixTreeAcceptor(self.alphabet)
        current_dfa = pta_builder.build_from_samples(positive_samples)
        pta_state_count = len(current_dfa.states)

        # Canonical ordering of states
        state_list = sorted(list(current_dfa.states), key=lambda x: (len(str(x)), str(x)))
        root_state = current_dfa.start_state

        RED: Set[State] = {root_state}
        trace_log: List[RPNITraceEntry] = []
        step_counter = 1
        merges_attempted = 0
        merges_accepted = 0

        def get_blue_states(curr_dfa: DFA, red_set: Set[State]) -> List[State]:
            blue = []
            for r in sorted(list(red_set), key=lambda x: (len(str(x)), str(x))):
                for sym in self.alphabet:
                    nxt = curr_dfa.transition(r, sym)
                    if nxt is not None and nxt not in red_set and nxt not in blue:
                        blue.append(nxt)
            return blue

        BLUE = get_blue_states(current_dfa, RED)

        while BLUE:
            q_b = BLUE.pop(0)
            merged = False

            # Try merging q_b with each q_r in RED in canonical order
            for q_r in sorted(list(RED), key=lambda x: (len(str(x)), str(x))):
                merges_attempted += 1
                candidate = self._fold(current_dfa, q_r, q_b)

                if candidate is not None:
                    consistent, violated_string = self._is_consistent(candidate, negative_samples)
                    if consistent:
                        # Successful merge!
                        current_dfa = candidate
                        merges_accepted += 1
                        merged = True
                        trace_log.append(RPNITraceEntry(
                            step_num=step_counter,
                            blue_state=q_b,
                            red_state=q_r,
                            action="MERGE_SUCCESS",
                            reason=f"Consistent merge with state {q_r}",
                            num_states_after=len(current_dfa.states),
                        ))
                        step_counter += 1
                        # Recompute RED (representatives in new DFA) and BLUE
                        RED = {s for s in RED if s in current_dfa.states}
                        if current_dfa.start_state not in RED:
                            RED.add(current_dfa.start_state)
                        BLUE = get_blue_states(current_dfa, RED)
                        break
                    else:
                        trace_log.append(RPNITraceEntry(
                            step_num=step_counter,
                            blue_state=q_b,
                            red_state=q_r,
                            action="MERGE_REJECTED",
                            reason=f"Violated negative sample '{violated_string}'",
                            num_states_after=len(current_dfa.states),
                        ))
                        step_counter += 1

            if not merged:
                # Promote q_b to RED
                RED.add(q_b)
                trace_log.append(RPNITraceEntry(
                    step_num=step_counter,
                    blue_state=q_b,
                    red_state=None,
                    action="PROMOTED",
                    reason=f"Cannot merge with any existing core state; promoted to core (RED)",
                    num_states_after=len(current_dfa.states),
                ))
                step_counter += 1
                BLUE = get_blue_states(current_dfa, RED)

        # Complete and minimize the learned DFA
        completed_dfa = current_dfa.complete("q_sink")
        min_result = minimize_dfa(completed_dfa)
        final_dfa = min_result.minimized_dfa.rename_states("q")

        t1 = time.perf_counter()
        exec_time_ms = (t1 - t0) * 1000.0

        return RPNIResult(
            learned_dfa=current_dfa.rename_states("p"),
            minimized_dfa=final_dfa,
            pta_state_count=pta_state_count,
            final_state_count=len(final_dfa.states),
            merges_attempted=merges_attempted,
            merges_accepted=merges_accepted,
            execution_time_ms=exec_time_ms,
            trace_log=trace_log,
        )
