"""Unit tests for dataset generation, noise injection, and CSV import/export."""

import pandas as pd
import pytest
from automatalearn.core.languages import get_language
from automatalearn.data.generator import (
    DataGenerator,
    export_dataset_to_csv,
    load_dataset_from_csv,
)


def test_data_generator_splits_and_balance():
    """Verify generated dataset splits and balance."""
    lang = get_language("ending_ab")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, dfa=lang.dfa, seed=42)

    splits = gen.generate_dataset(
        min_length=0,
        max_length=6,
        num_samples=100,
        class_balance=0.5,
        gen_min_length=7,
        gen_max_length=10,
        gen_samples=40,
    )

    assert len(splits.train_df) > 0
    assert len(splits.val_df) > 0
    assert len(splits.test_df) > 0
    assert len(splits.generalization_df) > 0

    # Ensure length of generalization set is strictly greater than max_length
    assert splits.generalization_df["length"].min() >= 7

    # Ensure labels match ground-truth oracle for test and generalization sets
    for s, y in zip(splits.test_strings, splits.test_labels):
        assert lang.predicate(s) == (y == 1)

    for s, y in zip(splits.gen_strings, splits.gen_labels):
        assert lang.predicate(s) == (y == 1)


def test_data_generator_noise_injection():
    """Verify noise flipping rate."""
    lang = get_language("contains_101")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, seed=42)

    samples = [("101", 1), ("000", 0), ("111", 0), ("1010", 1)] * 50
    noisy = gen.inject_noise(samples, noise_rate=0.2)

    flips = sum(1 for (s, y_orig), (_, y_noisy) in zip(samples, noisy) if y_orig != y_noisy)
    observed_rate = flips / len(samples)
    assert 0.10 <= observed_rate <= 0.30


def test_csv_export_import():
    """Verify round-trip CSV serialization."""
    df_orig = pd.DataFrame({
        "string": ["", "01", "101", "1111"],
        "label": [0, 1, 1, 0],
    })
    csv_str = export_dataset_to_csv(df_orig)
    df_loaded = load_dataset_from_csv(csv_str)

    assert len(df_loaded) == 4
    assert df_loaded["string"].tolist() == ["", "01", "101", "1111"]
    assert df_loaded["label"].tolist() == [0, 1, 1, 0]


def test_data_generator_default_large_sizes():
    """Verify default generation yields >= 500 in-distribution test strings and >= 500 gen strings."""
    lang = get_language("ending_ab")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, dfa=lang.dfa, seed=42)
    splits = gen.generate_dataset()
    assert len(splits.test_df) >= 500
    assert len(splits.generalization_df) >= 500
