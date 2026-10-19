"""Prefix Tree Acceptor (PTA) construction.

Builds tree-shaped Deterministic Finite Automata from positive and negative
training samples, establishing the initial state space for state-merging algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from automatalearn.core.dfa import DFA, State


@dataclass
class PTANode:
    """Node in the Prefix Tree Acceptor."""
    state_id: int
    prefix: str
    is_accept: bool = False
    children: Dict[str, PTANode] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}


class PrefixTreeAcceptor:
    """Builds a Prefix Tree Acceptor (PTA) from positive sample strings."""

    def __init__(self, alphabet: List[str]) -> None:
        self.alphabet = sorted(list(alphabet))
        self.root = PTANode(state_id=0, prefix="")
        self._next_id = 1
        self.nodes_by_id: Dict[int, PTANode] = {0: self.root}
        self.nodes_by_prefix: Dict[str, PTANode] = {"": self.root}

    def add_sample(self, s: str, is_accept: bool = True) -> PTANode:
        """Insert a sample string into the tree."""
        curr = self.root
        accum = ""
        for char in s:
            accum += char
            if char not in curr.children:
                new_node = PTANode(state_id=self._next_id, prefix=accum)
                self._next_id += 1
                curr.children[char] = new_node
                self.nodes_by_id[new_node.state_id] = new_node
                self.nodes_by_prefix[accum] = new_node
            curr = curr.children[char]

        if is_accept:
            curr.is_accept = True
        return curr

    def build_from_samples(self, positive_samples: List[str]) -> DFA:
        """Construct a DFA instance representing the Prefix Tree Acceptor."""
        for s in positive_samples:
            self.add_sample(s, is_accept=True)

        states: Set[State] = set(self.nodes_by_id.keys())
        accept_states: Set[State] = {
            nid for nid, node in self.nodes_by_id.items() if node.is_accept
        }
        transitions: Dict[Tuple[State, str], State] = {}

        for nid, node in self.nodes_by_id.items():
            for sym, child in node.children.items():
                transitions[(nid, sym)] = child.state_id

        return DFA(
            states=states,
            alphabet=set(self.alphabet),
            transitions=transitions,
            start_state=0,
            accept_states=accept_states,
            name="PrefixTreeAcceptor",
        )
