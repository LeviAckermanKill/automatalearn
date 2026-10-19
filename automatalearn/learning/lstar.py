"""Angluin's L* Active Learning Algorithm for Regular Languages.

Learns a minimal DFA using a Minimal Adequate Teacher (MAT) answering
Membership Queries (MQ) and Equivalence Queries (EQ) via an Observation Table.
Supports exact product-construction equivalence testing and approximate random sampling.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from automatalearn.core.dfa import DFA, State
from automatalearn.core.minimize import minimize_dfa


@dataclass
class LStarIterationLog:
    """Snapshot of an L* learning iteration."""
    iteration: int
    num_prefixes_S: int
    num_suffixes_E: int
    action: str
    detail: str
    membership_queries_so_far: int
    hypothesis_states: Optional[int] = None
    counterexample: Optional[str] = None


@dataclass
class LStarResult:
    """Output summary of L* learning."""
    learned_dfa: DFA
    minimized_dfa: DFA
    num_membership_queries: int
    num_equivalence_queries: int
    num_hypotheses: int
    execution_time_ms: float
    is_equivalent_to_target: bool
    observation_table_prefixes: List[str]
    observation_table_suffixes: List[str]
    trace_log: List[LStarIterationLog] = field(default_factory=list)

    @property
    def final_S(self) -> List[str]:
        """Convenience property for observation table prefixes (S)."""
        return self.observation_table_prefixes

    @property
    def final_E(self) -> List[str]:
        """Convenience property for observation table suffixes (E)."""
        return self.observation_table_suffixes


class LStarLearner:
    """Implements Angluin's L* active learning algorithm."""

    def __init__(
        self,
        alphabet: List[str],
        membership_oracle: Callable[[str], bool],
        equivalence_oracle: Optional[Callable[[DFA], Tuple[bool, Optional[str]]]] = None,
        target_dfa: Optional[DFA] = None,
        max_iterations: int = 100,
    ) -> None:
        """Initialize L* Learner.

        Args:
            alphabet: Target alphabet symbols.
            membership_oracle: Function answering string membership (MQ).
            equivalence_oracle: Optional custom equivalence test.
            target_dfa: Target DFA used for exact equivalence testing if equivalence_oracle is None.
            max_iterations: Safety ceiling to prevent infinite loops.
        """
        self.alphabet = sorted(list(alphabet))
        self.membership_oracle = membership_oracle
        self.target_dfa = target_dfa
        self.max_iterations = max_iterations

        if equivalence_oracle is not None:
            self.equivalence_oracle = equivalence_oracle
        elif target_dfa is not None:
            self.equivalence_oracle = lambda hyp: target_dfa.equivalent_to(hyp)
        else:
            self.equivalence_oracle = self._approximate_equivalence_testing

        self.mq_cache: Dict[str, bool] = {}
        self.num_mq = 0
        self.num_eq = 0

    def _query_mq(self, s: str) -> bool:
        """Query membership oracle with memoization caching."""
        if s not in self.mq_cache:
            self.num_mq += 1
            self.mq_cache[s] = self.membership_oracle(s)
        return self.mq_cache[s]

    def _approximate_equivalence_testing(self, hyp: DFA) -> Tuple[bool, Optional[str]]:
        """Fallback approximate equivalence test by random string sampling."""
        import random
        rng = random.Random(42)
        # Test short exhaustive strings first
        for length in range(0, 7):
            queue = [""]
            while queue:
                curr = queue.pop(0)
                if len(curr) == length:
                    if hyp.accepts(curr) != self._query_mq(curr):
                        return False, curr
                elif len(curr) < length:
                    for c in self.alphabet:
                        queue.append(curr + c)

        # Random sampling up to length 15
        for _ in range(500):
            length = rng.randint(1, 15)
            s = "".join(rng.choices(self.alphabet, k=length))
            if hyp.accepts(s) != self._query_mq(s):
                return False, s

        return True, None

    def learn(self) -> LStarResult:
        """Run L* algorithm and return learned DFA with full trace log."""
        t0 = time.perf_counter()
        self.mq_cache.clear()
        self.num_mq = 0
        self.num_eq = 0

        # S: prefixes (state identifiers), initially {""}
        S: List[str] = [""]
        # E: suffixes (distinguishing experiments), initially {""}
        E: List[str] = [""]
        # Table T: string -> row tuple of booleans for e in E
        # Keys in table are strings in S U (S . Sigma)

        trace_log: List[LStarIterationLog] = []
        iteration = 0
        num_hypotheses = 0

        def get_row(prefix: str) -> Tuple[int, ...]:
            return tuple(1 if self._query_mq(prefix + e) else 0 for e in E)

        while iteration < self.max_iterations:
            iteration += 1

            # Step 1: Ensure Closedness
            # Closed: for all t in S . Sigma, exists s in S such that row(t) == row(s)
            closed = True
            S_set = set(S)
            s_rows = {get_row(s) for s in S}

            for s in S:
                for a in self.alphabet:
                    t = s + a
                    if t not in S_set:
                        r_t = get_row(t)
                        if r_t not in s_rows:
                            # Not closed! Add t to S
                            S.append(t)
                            S_set.add(t)
                            closed = False
                            trace_log.append(LStarIterationLog(
                                iteration=iteration,
                                num_prefixes_S=len(S),
                                num_suffixes_E=len(E),
                                action="CLOSEDNESS_FIX",
                                detail=f"Added prefix '{t}' to S (unseen row {r_t})",
                                membership_queries_so_far=self.num_mq,
                            ))
                            break
                if not closed:
                    break

            if not closed:
                continue

            # Step 2: Ensure Consistency
            # Consistent: for all s1, s2 in S with row(s1) == row(s2), for all a in Sigma, row(s1.a) == row(s2.a)
            consistent = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    s1, s2 = S[i], S[j]
                    if get_row(s1) == get_row(s2):
                        for a in self.alphabet:
                            r1 = get_row(s1 + a)
                            r2 = get_row(s2 + a)
                            if r1 != r2:
                                # Find suffix e in E that separates s1.a and s2.a
                                for idx_e, e in enumerate(E):
                                    if r1[idx_e] != r2[idx_e]:
                                        new_e = a + e
                                        if new_e not in E:
                                            E.append(new_e)
                                            consistent = False
                                            trace_log.append(LStarIterationLog(
                                                iteration=iteration,
                                                num_prefixes_S=len(S),
                                                num_suffixes_E=len(E),
                                                action="CONSISTENCY_FIX",
                                                detail=f"Distinguished '{s1}' and '{s2}' by adding suffix '{new_e}' to E",
                                                membership_queries_so_far=self.num_mq,
                                            ))
                                            break
                                if not consistent:
                                    break
                    if not consistent:
                        break
                if not consistent:
                    break

            if not consistent:
                continue

            # Step 3: Table is Closed and Consistent -> Build Hypothesis DFA
            # Map unique rows to canonical state representatives
            row_to_state: Dict[Tuple[int, ...], str] = {}
            for s in S:
                r = get_row(s)
                if r not in row_to_state:
                    row_to_state[r] = s

            states = set(row_to_state.values())
            start_state = row_to_state[get_row("")]
            accept_states = {
                rep for rep in states
                if self._query_mq(rep)
            }

            transitions: Dict[Tuple[State, str], State] = {}
            for rep in states:
                for a in self.alphabet:
                    target_row = get_row(rep + a)
                    transitions[(rep, a)] = row_to_state[target_row]

            hypothesis_dfa = DFA(
                states=states,
                alphabet=set(self.alphabet),
                transitions=transitions,
                start_state=start_state,
                accept_states=accept_states,
                name=f"LStar_Hypothesis_{num_hypotheses}",
            ).rename_states("q")

            num_hypotheses += 1
            self.num_eq += 1

            # Step 4: Issue Equivalence Query
            is_equiv, counterexample = self.equivalence_oracle(hypothesis_dfa)

            if is_equiv or counterexample is None:
                # Successfully learned exact target DFA!
                trace_log.append(LStarIterationLog(
                    iteration=iteration,
                    num_prefixes_S=len(S),
                    num_suffixes_E=len(E),
                    action="EQUIVALENCE_CONFIRMED",
                    detail=f"Hypothesis with {len(hypothesis_dfa.states)} states is equivalent to target!",
                    membership_queries_so_far=self.num_mq,
                    hypothesis_states=len(hypothesis_dfa.states),
                ))
                break
            else:
                trace_log.append(LStarIterationLog(
                    iteration=iteration,
                    num_prefixes_S=len(S),
                    num_suffixes_E=len(E),
                    action="COUNTEREXAMPLE_RECEIVED",
                    detail=f"Counterexample received: '{counterexample}'",
                    membership_queries_so_far=self.num_mq,
                    hypothesis_states=len(hypothesis_dfa.states),
                    counterexample=counterexample,
                ))
                # Add counterexample and all its prefixes to S
                accum = ""
                for ch in counterexample:
                    accum += ch
                    if accum not in S_set:
                        S.append(accum)
                        S_set.add(accum)

        # Minimize final DFA
        min_result = minimize_dfa(hypothesis_dfa)
        final_minimized = min_result.minimized_dfa.rename_states("q")

        t1 = time.perf_counter()
        exec_time_ms = (t1 - t0) * 1000.0

        # Verify equivalence to target if available
        equiv_target = True
        if self.target_dfa is not None:
            equiv_target, _ = final_minimized.equivalent_to(self.target_dfa)

        return LStarResult(
            learned_dfa=hypothesis_dfa,
            minimized_dfa=final_minimized,
            num_membership_queries=self.num_mq,
            num_equivalence_queries=self.num_eq,
            num_hypotheses=num_hypotheses,
            execution_time_ms=exec_time_ms,
            is_equivalent_to_target=equiv_target,
            observation_table_prefixes=S,
            observation_table_suffixes=E,
            trace_log=trace_log,
        )
