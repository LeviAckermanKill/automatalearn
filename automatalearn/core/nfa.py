"""Nondeterministic Finite Automaton (NFA) implementation.

Supports epsilon transitions, step execution, acceptance testing,
and conversion to DFA via subset (powerset) construction.
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from automatalearn.core.dfa import DFA, State

EPSILON_SYMBOLS = {"", "ε", "\\e", "eps"}


class NFA:
    """Nondeterministic Finite Automaton with optional epsilon transitions.

    Formally defined as (Q, Sigma, delta, Q0, F):
      - Q: finite set of states
      - Sigma: alphabet
      - delta: transition function Q x (Sigma U {epsilon}) -> P(Q)
      - Q0: initial state(s) (subset of Q)
      - F: accept states (subset of Q)
    """

    def __init__(
        self,
        states: Set[State],
        alphabet: Set[str],
        transitions: Dict[Tuple[State, str], Set[State]],
        start_states: Union[State, Set[State]],
        accept_states: Set[State],
        name: Optional[str] = None,
    ) -> None:
        """Initialize an NFA.

        Args:
            states: Set of all state identifiers.
            alphabet: Alphabet of input characters (excluding epsilon).
            transitions: Dict mapping (state, symbol_or_epsilon) -> set of next states.
            start_states: Single state or set of starting states.
            accept_states: Set of accept states.
            name: Optional descriptive label.
        """
        self.states: Set[State] = set(states)
        self.alphabet: Set[str] = {c for c in alphabet if c not in EPSILON_SYMBOLS}
        self.transitions: Dict[Tuple[State, str], Set[State]] = {}
        for (st, sym), dsts in transitions.items():
            norm_sym = "" if sym in EPSILON_SYMBOLS else sym
            self.transitions[(st, norm_sym)] = set(dsts)
            self.states.add(st)
            self.states.update(dsts)
            if norm_sym != "":
                self.alphabet.add(norm_sym)

        if isinstance(start_states, (str, int)):
            self.start_states: Set[State] = {start_states}
        else:
            self.start_states = set(start_states)
        
        self.states.update(self.start_states)
        self.accept_states: Set[State] = set(accept_states)
        self.states.update(self.accept_states)
        self.name: Optional[str] = name

    def epsilon_closure(self, states: Set[State]) -> Set[State]:
        """Compute the epsilon-closure of a set of states via BFS."""
        closure = set(states)
        queue = deque(states)

        while queue:
            curr = queue.popleft()
            # Check empty string and all epsilon aliases
            eps_next = self.transitions.get((curr, ""), set())
            for nxt in eps_next:
                if nxt not in closure:
                    closure.add(nxt)
                    queue.append(nxt)

        return closure

    def step(self, states: Set[State], symbol: str) -> Set[State]:
        """Compute the set of states reached from `states` on `symbol`, including epsilon closure."""
        next_states: Set[State] = set()
        for st in states:
            targets = self.transitions.get((st, symbol), set())
            next_states.update(targets)
        return self.epsilon_closure(next_states)

    def accepts(self, s: str) -> bool:
        """Check whether the NFA accepts the input string."""
        curr = self.epsilon_closure(self.start_states)
        for char in s:
            if char not in self.alphabet:
                return False
            curr = self.step(curr, char)
            if not curr:
                return False
        return bool(curr & self.accept_states)

    def to_dfa(self, rename: bool = True) -> DFA:
        """Convert this NFA to an equivalent DFA using subset (powerset) construction.

        Args:
            rename: If True, rename states to canonical q0, q1, ... names.

        Returns:
            An equivalent complete or minimal-accessible DFA.
        """
        initial_subset = frozenset(self.epsilon_closure(self.start_states))
        unprocessed = deque([initial_subset])
        visited: Set[FrozenSet[State]] = {initial_subset}

        dfa_transitions: Dict[Tuple[FrozenSet[State], str], FrozenSet[State]] = {}
        dfa_states: Set[FrozenSet[State]] = {initial_subset}
        dfa_accept: Set[FrozenSet[State]] = set()

        if initial_subset & self.accept_states:
            dfa_accept.add(initial_subset)

        sorted_alphabet = sorted(self.alphabet)

        while unprocessed:
            curr_subset = unprocessed.popleft()
            for sym in sorted_alphabet:
                # Direct transitions from curr_subset on sym, followed by epsilon closure
                target_states: Set[State] = set()
                for st in curr_subset:
                    target_states.update(self.transitions.get((st, sym), set()))
                target_subset = frozenset(self.epsilon_closure(target_states))

                dfa_transitions[(curr_subset, sym)] = target_subset
                dfa_states.add(target_subset)

                if target_subset & self.accept_states:
                    dfa_accept.add(target_subset)

                if target_subset not in visited:
                    visited.add(target_subset)
                    unprocessed.append(target_subset)

        dfa = DFA(
            states=dfa_states,
            alphabet=self.alphabet,
            transitions=dfa_transitions,
            start_state=initial_subset,
            accept_states=dfa_accept,
            name=f"{self.name or 'NFA'}_determinized",
        )

        if rename:
            return dfa.rename_states()
        return dfa

    def to_dot(
        self,
        highlight_states: Optional[Set[State]] = None,
        title: Optional[str] = None,
    ) -> str:
        """Generate DOT language representation of this NFA."""
        from automatalearn.viz.draw import nfa_to_dot
        return nfa_to_dot(self, highlight_states=highlight_states, title=title)

    def to_graphviz(
        self,
        highlight_states: Optional[Set[State]] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Generate graphviz.Digraph representation of this NFA."""
        from automatalearn.viz.draw import nfa_to_graphviz
        return nfa_to_graphviz(self, highlight_states=highlight_states, title=title)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize NFA to a dictionary for JSON/UI."""
        trans_list = []
        for (st, sym), dsts in sorted(self.transitions.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            for dst in sorted(dsts, key=str):
                trans_list.append({"from": str(st), "symbol": sym if sym else "ε", "to": str(dst)})
        return {
            "name": self.name,
            "states": [str(s) for s in sorted(self.states, key=str)],
            "alphabet": sorted(self.alphabet),
            "start_states": [str(s) for s in sorted(self.start_states, key=str)],
            "accept_states": [str(s) for s in sorted(self.accept_states, key=str)],
            "transitions": trans_list,
        }

    def __repr__(self) -> str:
        return (
            f"NFA(states={len(self.states)}, alphabet={sorted(self.alphabet)}, "
            f"start={len(self.start_states)}, accept={len(self.accept_states)})"
        )
