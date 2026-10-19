"""Unit tests for string feature extraction."""

import numpy as np
import pytest
from automatalearn.ml.features import StringFeatureExtractor, FEATURE_GROUPS


def test_feature_extractor_basic_and_parity():
    """Verify basic lengths, symbol counts, and modular features."""
    alphabet = ["0", "1"]
    extractor = StringFeatureExtractor(alphabet=alphabet, enabled_groups=["basic", "parity"])
    train_strings = ["0", "1", "00", "01", "10", "11", "101", ""]
    extractor.fit(train_strings)

    feat_names = extractor.get_feature_names()
    assert "length" in feat_names
    assert "count_0" in feat_names
    assert "count_1" in feat_names
    assert "parity_0_mod2" in feat_names
    assert "parity_1_mod2" in feat_names
    assert "parity_0_mod3" in feat_names

    X = extractor.transform(["101", ""])
    assert X.shape == (2, len(feat_names))

    # Check "101" features: length=3, count_0=1, count_1=2, parity_0_mod2=1, parity_1_mod2=0
    idx_len = feat_names.index("length")
    idx_cnt0 = feat_names.index("count_0")
    idx_cnt1 = feat_names.index("count_1")
    idx_p0 = feat_names.index("parity_0_mod2")
    idx_p1 = feat_names.index("parity_1_mod2")

    assert X[0, idx_len] == 3.0
    assert X[0, idx_cnt0] == 1.0
    assert X[0, idx_cnt1] == 2.0
    assert X[0, idx_p0] == 1.0
    assert X[0, idx_p1] == 0.0

    # Check empty string
    assert X[1, idx_len] == 0.0


def test_feature_extractor_affixes_and_ngrams():
    """Verify prefix, suffix, and n-gram extraction."""
    alphabet = ["a", "b"]
    extractor = StringFeatureExtractor(alphabet=alphabet, enabled_groups=["affixes", "ngrams"])
    train_strings = ["ab", "aab", "bab", "b"]
    extractor.fit(train_strings)

    feat_names = extractor.get_feature_names()
    assert any("prefix" in f for f in feat_names)
    assert any("suffix" in f for f in feat_names)
    assert any("ngram_cnt_ab" in f for f in feat_names)

    X = extractor.transform(["ab", "ba"])
    assert X.shape[0] == 2
    assert not np.isnan(X).any()


def test_feature_extractor_runs():
    """Verify run-length calculation."""
    alphabet = ["0", "1"]
    extractor = StringFeatureExtractor(alphabet=alphabet, enabled_groups=["runs"])
    extractor.fit(["00011", "0101"])
    feat_names = extractor.get_feature_names()
    assert "max_run_0" in feat_names
    assert "num_alternations" in feat_names

    X = extractor.transform(["00011", "0101"])
    idx_max0 = feat_names.index("max_run_0")
    idx_alt = feat_names.index("num_alternations")

    assert X[0, idx_max0] == 3.0
    assert X[0, idx_alt] == 1.0  # single switch from 0 to 1

    assert X[1, idx_max0] == 1.0
    assert X[1, idx_alt] == 3.0  # 3 switches: 0->1->0->1


def test_feature_extractor_generic_vs_oracle():
    """Verify generic groups do not include oracle features, and oracle group extracts predicates."""
    from automatalearn.ml.features import GENERIC_FEATURE_GROUPS, ORACLE_FEATURE_GROUP

    extractor_generic = StringFeatureExtractor(alphabet=["a", "b"], enabled_groups=GENERIC_FEATURE_GROUPS)
    extractor_generic.fit(["ab", "ba"])
    names_generic = extractor_generic.get_feature_names()
    assert not any("oracle" in name for name in names_generic)
    assert "binary_mod3" not in names_generic

    extractor_oracle = StringFeatureExtractor(alphabet=["a", "b"], enabled_groups=[ORACLE_FEATURE_GROUP])
    extractor_oracle.fit(["ab", "ba"])
    names_oracle = extractor_oracle.get_feature_names()
    assert "oracle_suffix_ab" in names_oracle
    assert "oracle_contains_101" in names_oracle

    X_oracle = extractor_oracle.transform(["ab", "ba"])
    idx_suff = names_oracle.index("oracle_suffix_ab")
    assert X_oracle[0, idx_suff] == 1.0
    assert X_oracle[1, idx_suff] == 0.0


def test_feature_extractor_basic_only():
    """Verify Basic-only features contain strictly no mod-k, affixes, or oracle features."""
    from automatalearn.ml.features import BASIC_FEATURE_GROUPS

    assert "parity" not in BASIC_FEATURE_GROUPS
    assert "affixes" not in BASIC_FEATURE_GROUPS
    assert "position_weighted" not in BASIC_FEATURE_GROUPS
    assert "oracle" not in BASIC_FEATURE_GROUPS

    extractor = StringFeatureExtractor(alphabet=["0", "1"], enabled_groups=BASIC_FEATURE_GROUPS)
    extractor.fit(["0", "1", "00", "01", "10", "11"])
    names = extractor.get_feature_names()

    # Must contain length, counts, ngrams, runs
    assert "length" in names
    assert "count_0" in names
    assert "count_1" in names
    assert any("ngram" in n for n in names)
    assert any("max_run" in n for n in names)

    # Must NOT contain any mod-k parity or affixes or oracle
    assert not any("mod" in n for n in names)
    assert not any("parity" in n for n in names)
    assert not any("prefix" in n for n in names)
    assert not any("suffix" in n for n in names)
    assert not any("oracle" in n for n in names)

