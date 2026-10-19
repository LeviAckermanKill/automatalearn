"""Unit tests for DFA minimization (Hopcroft and Table-Filling)."""

import pytest
from automatalearn.core.dfa import DFA
from automatalearn.core.minimize import (
    minimize_dfa,
    minimize_dfa_hopcroft,
    minimize_dfa_table_filling,
)
from automatalearn.core.languages import get_language


def test_redundant_states_minimization():
    """Test that duplicate equivalent states are merged."""
    # Build DFA with redundant accept states: q1 and q2 are equivalent
    states = {"q0", "q1", "q2"}
    alphabet = {"a", "b"}
    transitions = {
        ("q0", "a"): "q1",
        ("q0", "b"): "q2",
        ("q1", "a"): "q1",
        ("q1", "b"): "q2",
        ("q2", "a"): "q1",
        ("q2", "b"): "q2",
    }
    # q1 and q2 both accept any subsequent string
    dfa = DFA(states, alphabet, transitions, "q0", {"q1", "q2"})

    res_hop = minimize_dfa_hopcroft(dfa)
    assert res_hop.before_state_count == 3
    assert res_hop.after_state_count == 2
    assert res_hop.minimized_dfa.accepts("a")
    assert res_hop.minimized_dfa.accepts("b")
    assert not res_hop.minimized_dfa.accepts("")

    res_tf = minimize_dfa_table_filling(dfa)
    assert res_tf.after_state_count == 2
    equiv, ce = res_hop.minimized_dfa.equivalent_to(res_tf.minimized_dfa)
    assert equiv is True


def test_minimization_preserves_language_all_benchmarks():
    """Verify minimization on all 7 benchmark languages preserves exact equivalence."""
    from automatalearn.core.languages import PREDEFINED_LANGUAGES

    for lang_key in PREDEFINED_LANGUAGES.keys():
        lang = get_language(lang_key)
        original = lang.dfa
        min_result = minimize_dfa(original)
        minimized = min_result.minimized_dfa

        # Check equivalence
        equiv, counterexample = original.equivalent_to(minimized)
        assert equiv is True, (
            f"Minimization altered language '{lang.name}'! Counterexample: '{counterexample}'"
        )
        assert min_result.after_state_count <= min_result.before_state_count


def test_table_filling_distinguishability_table():
    """Verify the distinguishability table produces correct marks."""
    lang = get_language("ending_ab")
    res = minimize_dfa_table_filling(lang.dfa)
    assert res.distinguishability_table is not None
    assert len(res.distinguishability_table) > 0


def test_non_minimal_example_dfa_reduction():
    """Verify the non-minimal example DFA reduces from 6 states to 3 states (-50%)."""
    from automatalearn.core.minimize import get_non_minimal_example_dfa

    dfa = get_non_minimal_example_dfa()
    assert len(dfa.states) == 6
    res_hop = minimize_dfa_hopcroft(dfa)
    assert res_hop.before_state_count == 6
    assert res_hop.after_state_count == 3
    assert res_hop.reduction_percentage == 50.0

    res_tf = minimize_dfa_table_filling(dfa)
    assert res_tf.before_state_count == 6
    assert res_tf.after_state_count == 3
    assert res_tf.reduction_percentage == 50.0

