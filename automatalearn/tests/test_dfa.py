"""Unit tests for DFA class."""

import pytest
from automatalearn.core.dfa import DFA
from automatalearn.core.languages import get_language, PREDEFINED_LANGUAGES


def test_dfa_basic_acceptance():
    """Test standard DFA accepts and rejects properly."""
    # Language: strings ending with 'ab'
    states = {"q0", "q1", "q2"}
    alphabet = {"a", "b"}
    transitions = {
        ("q0", "a"): "q1",
        ("q0", "b"): "q0",
        ("q1", "a"): "q1",
        ("q1", "b"): "q2",
        ("q2", "a"): "q1",
        ("q2", "b"): "q0",
    }
    dfa = DFA(states, alphabet, transitions, "q0", {"q2"})

    assert dfa.accepts("ab")
    assert dfa.accepts("aab")
    assert dfa.accepts("babab")
    assert not dfa.accepts("")
    assert not dfa.accepts("a")
    assert not dfa.accepts("b")
    assert not dfa.accepts("aba")


def test_dfa_trace():
    """Test state tracing for a string."""
    lang = get_language("ending_ab")
    dfa = lang.dfa

    acc, path = dfa.trace("aab")
    assert acc is True
    assert path == ["q0", "q1", "q1", "q2"]

    acc_rej, path_rej = dfa.trace("aa")
    assert acc_rej is False
    assert path_rej == ["q0", "q1", "q1"]


def test_dfa_completion_and_unreachable():
    """Test DFA completion and unreachable state pruning."""
    # Incomplete DFA with unreachable state 'q_ghost'
    states = {"q0", "q1", "q_ghost"}
    alphabet = {"0", "1"}
    transitions = {
        ("q0", "0"): "q1",
        # missing (q0, 1), (q1, 0), (q1, 1)
        ("q_ghost", "0"): "q0",
    }
    dfa = DFA(states, alphabet, transitions, "q0", {"q1"})

    assert not dfa.is_complete()
    assert "q_ghost" in dfa.states

    accessible = dfa.remove_unreachable()
    assert "q_ghost" not in accessible.states
    assert len(accessible.states) == 2

    completed = accessible.complete("q_sink")
    assert completed.is_complete()
    assert "q_sink" in completed.states
    assert completed.accepts("0")
    assert not completed.accepts("1")
    assert not completed.accepts("00")


def test_dfa_equivalence_product():
    """Test DFA equivalence testing and counterexample generation."""
    lang = get_language("contains_101")
    dfa1 = lang.dfa
    dfa2 = lang.dfa.copy()

    # Identical DFAs
    equiv, counterexample = dfa1.equivalent_to(dfa2)
    assert equiv is True
    assert counterexample is None

    # Alter dfa2 to reject "101"
    dfa2.accept_states = set()
    equiv2, counterexample2 = dfa1.equivalent_to(dfa2)
    assert equiv2 is False
    assert counterexample2 is not None
    # Counterexample must separate the two
    assert dfa1.accepts(counterexample2) != dfa2.accepts(counterexample2)


@pytest.mark.parametrize("lang_key", list(PREDEFINED_LANGUAGES.keys()))
def test_predefined_languages_consistency(lang_key: str):
    """Verify ground truth DFA matches ground truth predicate for sample strings."""
    lang = get_language(lang_key)
    alphabet = lang.alphabet

    # Test small universe of strings up to length 6
    test_strings = [""]
    for sym1 in alphabet:
        test_strings.append(sym1)
        for sym2 in alphabet:
            test_strings.append(sym1 + sym2)
            for sym3 in alphabet:
                test_strings.append(sym1 + sym2 + sym3)

    for s in test_strings:
        dfa_result = lang.dfa.accepts(s)
        pred_result = lang.predicate(s)
        assert dfa_result == pred_result, f"Mismatch on string '{s}' for language '{lang.name}'"
