"""Unit tests for NFA and Regex-to-NFA conversion."""

import pytest
from automatalearn.core.nfa import NFA
from automatalearn.core.regex_to_nfa import regex_to_nfa


def test_nfa_epsilon_closure_and_acceptance():
    """Test NFA with epsilon transitions."""
    # NFA accepting 'a' or 'b' via epsilon branches
    states = {"q_start", "q_a", "q_b", "q_f"}
    alphabet = {"a", "b"}
    transitions = {
        ("q_start", ""): {"q_a", "q_b"},
        ("q_a", "a"): {"q_f"},
        ("q_b", "b"): {"q_f"},
    }
    nfa = NFA(states, alphabet, transitions, "q_start", {"q_f"})

    assert nfa.epsilon_closure({"q_start"}) == {"q_start", "q_a", "q_b"}
    assert nfa.accepts("a")
    assert nfa.accepts("b")
    assert not nfa.accepts("")
    assert not nfa.accepts("ab")


def test_nfa_subset_construction():
    """Test converting NFA to DFA via powerset construction."""
    # NFA for strings ending with '01'
    states = {"s0", "s1", "s2"}
    alphabet = {"0", "1"}
    transitions = {
        ("s0", "0"): {"s0", "s1"},
        ("s0", "1"): {"s0"},
        ("s1", "1"): {"s2"},
    }
    nfa = NFA(states, alphabet, transitions, "s0", {"s2"})

    dfa = nfa.to_dfa()
    assert dfa.accepts("01")
    assert dfa.accepts("0001")
    assert dfa.accepts("1101")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("010")

    # Equivalence on multiple strings
    test_cases = ["", "0", "1", "00", "01", "10", "11", "001", "101", "0101", "1110"]
    for s in test_cases:
        assert nfa.accepts(s) == dfa.accepts(s)


def test_regex_to_nfa_thompson():
    """Test compiling regular expressions to NFA and determinizing."""
    # Language (a|b)*ab
    nfa = regex_to_nfa("(a|b)*ab")
    dfa = nfa.to_dfa()

    assert dfa.accepts("ab")
    assert dfa.accepts("aab")
    assert dfa.accepts("babab")
    assert not dfa.accepts("")
    assert not dfa.accepts("a")
    assert not dfa.accepts("ba")


def test_regex_complex():
    """Test regex with plus, optional, and nested parentheses."""
    nfa = regex_to_nfa("a+b?c")
    dfa = nfa.to_dfa()

    assert dfa.accepts("abc")
    assert dfa.accepts("ac")
    assert dfa.accepts("aaabc")
    assert not dfa.accepts("bc")
    assert not dfa.accepts("ab")
