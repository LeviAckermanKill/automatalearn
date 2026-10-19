"""DFA Minimization via Hopcroft's Partition Refinement and Table-Filling Algorithm.

Supports both Hopcroft's algorithm (O(k n log n)) and the Myhill-Nerode
Table-Filling algorithm, returning detailed before/after state metrics,
merged equivalence classes, and distinguishability matrices for visualization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from automatalearn.core.dfa import DFA, State


@dataclass
class MinimizationResult:
    """Detailed result of DFA minimization."""

    original_dfa: DFA
    minimized_dfa: DFA
    before_state_count: int
    after_state_count: int
    equivalence_classes: List[Set[State]]
    method_used: str
    distinguishability_table: Optional[Dict[Tuple[State, State], Tuple[bool, Optional[str]]]] = None
    partition_history: List[List[Set[State]]] = field(default_factory=list)

    @property
    def reduction_percentage(self) -> float:
        """Percentage of states reduced, clamped to >= 0.0."""
        if self.before_state_count == 0:
            return 0.0
        pct = ((self.before_state_count - self.after_state_count) / self.before_state_count) * 100.0
        return max(0.0, float(pct))


def minimize_dfa_hopcroft(dfa: DFA) -> MinimizationResult:
    """Minimize a DFA using Hopcroft's algorithm.

    Args:
        dfa: Input DFA.

    Returns:
        MinimizationResult containing minimized DFA and equivalence classes.
    """
    # 1. Remove unreachable states
    accessible_dfa = dfa.remove_unreachable()
    # 2. Ensure DFA is complete
    complete_dfa = accessible_dfa.complete(dead_state="q_dead_sink")
    
    states = complete_dfa.states
    alphabet = sorted(complete_dfa.alphabet)
    accept = complete_dfa.accept_states
    non_accept = states - accept

    # If all states accept or no states accept, language is Sigma* or empty
    if not accept:
        single_state = "q_sink"
        sink_trans = {(single_state, a): single_state for a in alphabet}
        min_dfa = DFA(
            states={single_state},
            alphabet=set(alphabet),
            transitions=sink_trans,
            start_state=single_state,
            accept_states=set(),
            name=f"{dfa.name or 'DFA'}_minimized",
        )
        return MinimizationResult(
            original_dfa=dfa,
            minimized_dfa=min_dfa,
            before_state_count=len(dfa.states),
            after_state_count=1,
            equivalence_classes=[set(states)],
            method_used="Hopcroft",
            partition_history=[[set(states)]],
        )

    if not non_accept:
        single_state = "q_accept_all"
        trans = {(single_state, a): single_state for a in alphabet}
        min_dfa = DFA(
            states={single_state},
            alphabet=set(alphabet),
            transitions=trans,
            start_state=single_state,
            accept_states={single_state},
            name=f"{dfa.name or 'DFA'}_minimized",
        )
        return MinimizationResult(
            original_dfa=dfa,
            minimized_dfa=min_dfa,
            before_state_count=len(dfa.states),
            after_state_count=1,
            equivalence_classes=[set(states)],
            method_used="Hopcroft",
            partition_history=[[set(states)]],
        )

    # Initial partition P: {F, Q \ F}
    P: Set[FrozenSet[State]] = {frozenset(accept), frozenset(non_accept)}
    # Worklist W: pick smaller set
    W: Set[FrozenSet[State]] = {frozenset(accept) if len(accept) <= len(non_accept) else frozenset(non_accept)}

    # Precompute inverse transitions: for each symbol c and state q, which states lead to q
    inv_trans: Dict[Tuple[str, State], Set[State]] = {}
    for (st, sym), dst in complete_dfa.transitions.items():
        inv_trans.setdefault((sym, dst), set()).add(st)

    partition_history: List[List[Set[State]]] = [[set(b) for b in P]]

    while W:
        A = W.pop()
        for c in alphabet:
            # X = pre(A, c) = {q in Q | delta(q, c) in A}
            X: Set[State] = set()
            for q in A:
                X.update(inv_trans.get((c, q), set()))

            new_P: Set[FrozenSet[State]] = set()
            partition_changed = False

            for Y in P:
                Y_inter = Y.intersection(X)
                Y_diff = Y.difference(X)
                if Y_inter and Y_diff:
                    new_P.add(frozenset(Y_inter))
                    new_P.add(frozenset(Y_diff))
                    partition_changed = True

                    if Y in W:
                        W.remove(Y)
                        W.add(frozenset(Y_inter))
                        W.add(frozenset(Y_diff))
                    else:
                        if len(Y_inter) <= len(Y_diff):
                            W.add(frozenset(Y_inter))
                        else:
                            W.add(frozenset(Y_diff))
                else:
                    new_P.add(Y)

            P = new_P
            if partition_changed:
                partition_history.append([set(b) for b in P])

    # Convert partition blocks to equivalence classes
    equivalence_classes = [set(b) for b in P]

    # Map each state to its representative
    state_to_rep: Dict[State, State] = {}
    for block in P:
        # Choose a canonical representative (prioritize original start state if in block)
        if complete_dfa.start_state in block:
            rep = complete_dfa.start_state
        else:
            rep = sorted(list(block), key=str)[0]
        for s in block:
            state_to_rep[s] = rep

    new_states = {state_to_rep[s] for s in complete_dfa.states}
    new_start = state_to_rep[complete_dfa.start_state]
    new_accept = {state_to_rep[s] for s in complete_dfa.accept_states}
    new_transitions: Dict[Tuple[State, str], State] = {}

    for (st, sym), dst in complete_dfa.transitions.items():
        rep_st = state_to_rep[st]
        rep_dst = state_to_rep[dst]
        new_transitions[(rep_st, sym)] = rep_dst

    minimized_dfa = DFA(
        states=new_states,
        alphabet=complete_dfa.alphabet,
        transitions=new_transitions,
        start_state=new_start,
        accept_states=new_accept,
        name=f"{dfa.name or 'DFA'}_minimized",
    ).remove_unreachable().rename_states()

    return MinimizationResult(
        original_dfa=dfa,
        minimized_dfa=minimized_dfa,
        before_state_count=len(dfa.states),
        after_state_count=len(minimized_dfa.states),
        equivalence_classes=equivalence_classes,
        method_used="Hopcroft",
        partition_history=partition_history,
    )


def minimize_dfa_table_filling(dfa: DFA) -> MinimizationResult:
    """Minimize DFA using the Table-Filling (Myhill-Nerode) algorithm.

    Constructs a distinguishability table for all pairs {p, q}.
    Educational algorithm showing exactly which witness symbol separates each state pair.

    Args:
        dfa: Input DFA.

    Returns:
        MinimizationResult with the full distinguishability matrix.
    """
    accessible = dfa.remove_unreachable().complete(dead_state="q_sink")
    state_list = sorted(list(accessible.states), key=str)
    alphabet = sorted(accessible.alphabet)
    n = len(state_list)

    # dist_table[(p, q)] = (is_distinguished: bool, distinguishing_symbol: Optional[str])
    # Indexed such that index(p) < index(q)
    dist_table: Dict[Tuple[State, State], Tuple[bool, Optional[str]]] = {}

    def get_pair(s1: State, s2: State) -> Tuple[State, State]:
        return (s1, s2) if state_list.index(s1) < state_list.index(s2) else (s2, s1)

    # Step 1: Base case - mark pairs where exactly one is in accept_states (distinguished by epsilon)
    for i in range(n):
        for j in range(i + 1, n):
            p, q = state_list[i], state_list[j]
            p_acc = p in accessible.accept_states
            q_acc = q in accessible.accept_states
            if p_acc != q_acc:
                dist_table[(p, q)] = (True, "ε")
            else:
                dist_table[(p, q)] = (False, None)

    # Step 2: Inductive marking
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                p, q = state_list[i], state_list[j]
                if dist_table[(p, q)][0]:
                    continue  # already distinguished
                for sym in alphabet:
                    np = accessible.transition(p, sym)
                    nq = accessible.transition(q, sym)
                    if np is not None and nq is not None and np != nq:
                        pair = get_pair(np, nq)
                        if dist_table.get(pair, (False, None))[0]:
                            dist_table[(p, q)] = (True, sym)
                            changed = True
                            break

    # Step 3: Find equivalence classes by connecting unmarked pairs
    parent = {s: s for s in state_list}

    def find(s: State) -> State:
        if parent[s] != s:
            parent[s] = find(parent[s])
        return parent[s]

    def union(s1: State, s2: State) -> None:
        r1, r2 = find(s1), find(s2)
        if r1 != r2:
            parent[r2] = r1

    for (p, q), (distinguished, _) in dist_table.items():
        if not distinguished:
            union(p, q)

    class_dict: Dict[State, Set[State]] = {}
    for s in state_list:
        rep = find(s)
        class_dict.setdefault(rep, set()).add(s)

    equivalence_classes = list(class_dict.values())

    # Build minimized DFA
    new_states = {find(s) for s in accessible.states}
    new_start = find(accessible.start_state)
    new_accept = {find(s) for s in accessible.accept_states}
    new_transitions: Dict[Tuple[State, str], State] = {}

    for (st, sym), dst in accessible.transitions.items():
        rep_st = find(st)
        rep_dst = find(dst)
        new_transitions[(rep_st, sym)] = rep_dst

    minimized = DFA(
        states=new_states,
        alphabet=accessible.alphabet,
        transitions=new_transitions,
        start_state=new_start,
        accept_states=new_accept,
        name=f"{dfa.name or 'DFA'}_minimized",
    ).remove_unreachable().rename_states()

    return MinimizationResult(
        original_dfa=dfa,
        minimized_dfa=minimized,
        before_state_count=len(dfa.states),
        after_state_count=len(minimized.states),
        equivalence_classes=equivalence_classes,
        method_used="Table-Filling",
        distinguishability_table=dist_table,
    )


def minimize_dfa(dfa: DFA, method: str = "hopcroft") -> MinimizationResult:
    """Convenience entry point for minimizing a DFA.

    Args:
        dfa: Input DFA.
        method: "hopcroft" (default) or "table_filling".

    Returns:
        MinimizationResult.
    """
    if method.lower() in ("table_filling", "table", "myhill_nerode"):
        return minimize_dfa_table_filling(dfa)
    return minimize_dfa_hopcroft(dfa)


def get_non_minimal_example_dfa() -> DFA:
    """Return a textbook non-minimal 6-state DFA over alphabet {'0', '1'} that minimizes to 3 states.

    Contains redundant pairs {A, B}, {C, D}, and {E, F}.
    Hopcroft / Table-Filling merges them into 3 equivalence classes (50.0% state reduction).
    """
    states = {"A", "B", "C", "D", "E", "F"}
    alphabet = {"0", "1"}
    transitions = {
        ("A", "0"): "B", ("A", "1"): "C",
        ("B", "0"): "A", ("B", "1"): "D",
        ("C", "0"): "E", ("C", "1"): "F",
        ("D", "0"): "E", ("D", "1"): "F",
        ("E", "0"): "E", ("E", "1"): "F",
        ("F", "0"): "E", ("F", "1"): "F",
    }
    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="A",
        accept_states={"C", "D"},
        name="NonMinimal_6State_Example",
    )

