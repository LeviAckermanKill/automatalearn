"""Unit tests for RPNI and Angluin's L* learning algorithms."""

import pytest
from automatalearn.core.languages import get_language
from automatalearn.data.generator import DataGenerator
from automatalearn.learning.lstar import LStarLearner
from automatalearn.learning.prefix_tree import PrefixTreeAcceptor
from automatalearn.learning.rpni import RPNILearner


def test_prefix_tree_acceptor():
    """Verify PTA accepts all inserted positive samples."""
    alphabet = ["a", "b"]
    samples = ["a", "ab", "aab", "aba"]
    pta_builder = PrefixTreeAcceptor(alphabet)
    dfa = pta_builder.build_from_samples(samples)

    for s in samples:
        assert dfa.accepts(s)

    # Negative strings should not reach accepting states
    assert not dfa.accepts("")
    assert not dfa.accepts("b")
    assert not dfa.accepts("abb")


def test_rpni_consistency_with_training_data():
    """Verify RPNI outputs a DFA strictly consistent with all positive and negative samples."""
    lang = get_language("ending_ab")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, dfa=lang.dfa, seed=42)
    splits = gen.generate_dataset(min_length=0, max_length=5, num_samples=60)

    pos_samples = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 1]
    neg_samples = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 0]

    learner = RPNILearner(alphabet=lang.alphabet)
    res = learner.learn(pos_samples, neg_samples)

    assert res.pta_state_count > res.final_state_count
    assert res.merges_accepted > 0

    # Strict consistency check on training samples
    for p in pos_samples:
        assert res.minimized_dfa.accepts(p), f"RPNI DFA failed to accept positive sample '{p}'"
    for n in neg_samples:
        assert not res.minimized_dfa.accepts(n), f"RPNI DFA incorrectly accepted negative sample '{n}'"


@pytest.mark.parametrize("lang_key", ["ending_ab", "count_a_div_3", "contains_101"])
def test_lstar_exact_convergence(lang_key: str):
    """Verify L* converges to a DFA exactly equivalent to the target DFA."""
    lang = get_language(lang_key)
    target_dfa = lang.dfa

    learner = LStarLearner(
        alphabet=lang.alphabet,
        membership_oracle=lang.predicate,
        target_dfa=target_dfa,
    )
    result = learner.learn()

    assert result.is_equivalent_to_target is True
    equiv, counterexample = result.minimized_dfa.equivalent_to(target_dfa)
    assert equiv is True
    assert counterexample is None
    # Check that state count matches minimal target state count
    assert len(result.minimized_dfa.states) == len(target_dfa.states)
