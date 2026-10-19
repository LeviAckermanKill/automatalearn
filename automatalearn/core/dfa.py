"""Deterministic Finite Automaton (DFA) implementation.

Provides the DFA class with acceptance checking, path tracing, completeness
verification, dead state addition, unreachable state pruning, product-construction
equivalence testing with counterexample generation, and Graphviz visualization.
"""

from __future__ import annotations
from collections import deque
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

State = Union[str, int, Tuple[Any, ...], FrozenSet[Any]]


class DFA:
    """A Deterministic Finite Automaton (DFA).

    Formally defined as a 5-tuple (Q, Sigma, delta, q0, F):
      - Q: finite set of states
      - Sigma: finite alphabet of symbols
      - delta: transition function Q x Sigma -> Q
      - q0: initial / start state in Q
      - F: set of accept / final states (subset of Q)
    """

    def __init__(
        self,
        states: Set[State],
        alphabet: Set[str],
        transitions: Dict[Tuple[State, str], State],
        start_state: State,
        accept_states: Set[State],
        name: Optional[str] = None,
    ) -> None:
        """Initialize a DFA.

        Args:
            states: Set of all state identifiers.
            alphabet: Set of input symbols (each should be single character).
            transitions: Dict mapping (state, symbol) -> next_state.
            start_state: Initial state (must be in states).
            accept_states: Set of accepting states (must be subset of states).
            name: Optional descriptive label for this DFA.
        """
        self.states: Set[State] = set(states)
        self.alphabet: Set[str] = set(alphabet)
        self.transitions: Dict[Tuple[State, str], State] = dict(transitions)
        self.start_state: State = start_state
        self.accept_states: Set[State] = set(accept_states)
        self.name: Optional[str] = name

        # Validate basic properties
        if self.start_state not in self.states:
            self.states.add(self.start_state)
        
        # Ensure accept states are subset of states
        for acc in self.accept_states:
            if acc not in self.states:
                self.states.add(acc)

        # Ensure transition endpoints belong to states
        for (st, sym), dst in list(self.transitions.items()):
            self.states.add(st)
            self.states.add(dst)
            self.alphabet.add(sym)

    def transition(self, state: State, symbol: str) -> Optional[State]:
        """Return the next state from state on symbol, or None if undefined."""
        return self.transitions.get((state, symbol))

    def accepts(self, s: str) -> bool:
        """Determine whether the DFA accepts the input string.

        Args:
            s: Input string over the DFA's alphabet.

        Returns:
            True if the string reaches an accept state, False otherwise.
        """
        curr = self.start_state
        for char in s:
            if char not in self.alphabet:
                return False
            nxt = self.transition(curr, char)
            if nxt is None:
                return False
            curr = nxt
        return curr in self.accept_states

    def trace(self, s: str) -> Tuple[bool, List[State]]:
        """Trace the state transitions for the given input string.

        Args:
            s: Input string to trace.

        Returns:
            Tuple (accepted, path) where path is the list of states visited
            from start_state to the end (or until undefined transition).
        """
        path = [self.start_state]
        curr = self.start_state
        for char in s:
            if char not in self.alphabet:
                path.append("DEAD_CHAR")
                return False, path
            nxt = self.transition(curr, char)
            if nxt is None:
                path.append("DEAD_TRANSITION")
                return False, path
            curr = nxt
            path.append(curr)
        return (curr in self.accept_states), path

    def is_complete(self) -> bool:
        """Check whether every (state, symbol) pair has a defined transition."""
        for state in self.states:
            for sym in self.alphabet:
                if (state, sym) not in self.transitions:
                    return False
        return True

    def complete(self, dead_state: State = "q_dead") -> DFA:
        """Return a new complete DFA by routing missing transitions to a dead state.

        Args:
            dead_state: Identifier for the sink state.

        Returns:
            A new complete DFA.
        """
        if self.is_complete():
            return self.copy()

        new_states = set(self.states)
        new_transitions = dict(self.transitions)
        new_accept = set(self.accept_states)
        needs_dead = False

        for state in self.states:
            for sym in self.alphabet:
                if (state, sym) not in new_transitions:
                    new_transitions[(state, sym)] = dead_state
                    needs_dead = True

        if needs_dead:
            new_states.add(dead_state)
            for sym in self.alphabet:
                new_transitions[(dead_state, sym)] = dead_state

        return DFA(
            states=new_states,
            alphabet=self.alphabet,
            transitions=new_transitions,
            start_state=self.start_state,
            accept_states=new_accept,
            name=f"{self.name or 'DFA'}_completed",
        )

    def remove_unreachable(self) -> DFA:
        """Return a new DFA containing only states reachable from start_state."""
        reachable: Set[State] = set()
        queue: deque[State] = deque([self.start_state])
        reachable.add(self.start_state)

        while queue:
            curr = queue.popleft()
            for sym in self.alphabet:
                nxt = self.transition(curr, sym)
                if nxt is not None and nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

        new_transitions = {
            (st, sym): dst
            for (st, sym), dst in self.transitions.items()
            if st in reachable and dst in reachable
        }
        new_accept = self.accept_states.intersection(reachable)

        return DFA(
            states=reachable,
            alphabet=self.alphabet,
            transitions=new_transitions,
            start_state=self.start_state,
            accept_states=new_accept,
            name=self.name,
        )

    def equivalent_to(self, other: DFA) -> Tuple[bool, Optional[str]]:
        """Test equivalence between this DFA and another using product construction.

        Checks if the symmetric difference language L(self) XOR L(other) is empty.
        If non-empty, finds the shortest counterexample string via BFS.

        Args:
            other: The other DFA to compare with.

        Returns:
            Tuple (is_equivalent, counterexample). If equivalent, (True, None).
            If not equivalent, (False, counterexample_string).
        """
        # Ensure common alphabet
        common_alphabet = sorted(list(self.alphabet | other.alphabet))
        
        # Complete both DFAs over the common alphabet
        d1 = self._complete_with_alphabet(common_alphabet, "dead1").remove_unreachable()
        d2 = other._complete_with_alphabet(common_alphabet, "dead2").remove_unreachable()

        start_pair = (d1.start_state, d2.start_state)
        # Check start state immediately
        in_f1 = start_pair[0] in d1.accept_states
        in_f2 = start_pair[1] in d2.accept_states
        if in_f1 != in_f2:
            return False, ""  # empty string is counterexample

        visited: Set[Tuple[State, State]] = {start_pair}
        # queue stores ((q1, q2), path_string)
        queue: deque[Tuple[Tuple[State, State], str]] = deque([(start_pair, "")])

        while queue:
            (q1, q2), path = queue.popleft()
            for sym in common_alphabet:
                nxt1 = d1.transition(q1, sym)
                nxt2 = d2.transition(q2, sym)
                nxt_pair = (nxt1, nxt2)

                if nxt_pair not in visited:
                    visited.add(nxt_pair)
                    nxt_path = path + sym
                    acc1 = nxt1 in d1.accept_states
                    acc2 = nxt2 in d2.accept_states
                    if acc1 != acc2:
                        return False, nxt_path
                    queue.append((nxt_pair, nxt_path))

        return True, None

    def _complete_with_alphabet(self, target_alphabet: List[str], dead_state_name: str) -> DFA:
        """Internal helper: complete DFA over a specific superset alphabet."""
        new_states = set(self.states)
        new_transitions = dict(self.transitions)
        alphabet_set = set(target_alphabet)
        needs_dead = False

        for st in self.states:
            for sym in alphabet_set:
                if (st, sym) not in new_transitions:
                    new_transitions[(st, sym)] = dead_state_name
                    needs_dead = True

        if needs_dead:
            new_states.add(dead_state_name)
            for sym in alphabet_set:
                new_transitions[(dead_state_name, sym)] = dead_state_name

        return DFA(
            states=new_states,
            alphabet=alphabet_set,
            transitions=new_transitions,
            start_state=self.start_state,
            accept_states=set(self.accept_states),
        )

    def rename_states(self, prefix: str = "q") -> DFA:
        """Return an isomorphic DFA with canonical state names q0, q1, ... via BFS."""
        cleaned = self.remove_unreachable()
        mapping: Dict[State, str] = {cleaned.start_state: f"{prefix}0"}
        counter = 1
        queue = deque([cleaned.start_state])

        while queue:
            curr = queue.popleft()
            for sym in sorted(cleaned.alphabet):
                nxt = cleaned.transition(curr, sym)
                if nxt is not None and nxt not in mapping:
                    mapping[nxt] = f"{prefix}{counter}"
                    counter += 1
                    queue.append(nxt)

        new_states = {mapping[s] for s in cleaned.states}
        new_start = mapping[cleaned.start_state]
        new_accept = {mapping[s] for s in cleaned.accept_states if s in mapping}
        new_transitions = {
            (mapping[st], sym): mapping[dst]
            for (st, sym), dst in cleaned.transitions.items()
            if st in mapping and dst in mapping
        }

        return DFA(
            states=new_states,
            alphabet=cleaned.alphabet,
            transitions=new_transitions,
            start_state=new_start,
            accept_states=new_accept,
            name=self.name,
        )

    def copy(self) -> DFA:
        """Return a shallow-copy clone of this DFA."""
        return DFA(
            states=set(self.states),
            alphabet=set(self.alphabet),
            transitions=dict(self.transitions),
            start_state=self.start_state,
            accept_states=set(self.accept_states),
            name=self.name,
        )

    def to_dot(
        self,
        highlight_states: Optional[Set[State]] = None,
        active_path: Optional[List[State]] = None,
        title: Optional[str] = None,
    ) -> str:
        """Generate DOT language representation of this DFA."""
        from automatalearn.viz.draw import dfa_to_dot
        return dfa_to_dot(
            self,
            highlight_states=highlight_states,
            active_path=active_path,
            title=title,
        )

    def to_graphviz(
        self,
        highlight_states: Optional[Set[State]] = None,
        active_path: Optional[List[State]] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Generate a graphviz.Digraph object representing this DFA."""
        from automatalearn.viz.draw import dfa_to_graphviz
        return dfa_to_graphviz(
            self,
            highlight_states=highlight_states,
            active_path=active_path,
            title=title,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DFA to a dictionary for JSON/UI consumption."""
        trans_list = [
            {"from": str(st), "symbol": sym, "to": str(dst)}
            for (st, sym), dst in sorted(self.transitions.items(), key=lambda x: (str(x[0][0]), x[0][1]))
        ]
        return {
            "name": self.name,
            "states": [str(s) for s in sorted(self.states, key=str)],
            "alphabet": sorted(self.alphabet),
            "start_state": str(self.start_state),
            "accept_states": [str(s) for s in sorted(self.accept_states, key=str)],
            "transitions": trans_list,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DFA:
        """Construct DFA from a dictionary representation."""
        states = set(data["states"])
        alphabet = set(data["alphabet"])
        start_state = data["start_state"]
        accept_states = set(data["accept_states"])
        transitions: Dict[Tuple[State, str], State] = {}
        for item in data.get("transitions", []):
            transitions[(item["from"], item["symbol"])] = item["to"]
        return cls(
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            start_state=start_state,
            accept_states=accept_states,
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return (
            f"DFA(states={len(self.states)}, alphabet={sorted(self.alphabet)}, "
            f"start={self.start_state}, accept={len(self.accept_states)})"
        )
