"""Feature extraction pipeline converting string samples to numerical feature vectors.

Supports modular feature groups: basic lengths/counts/ratios, parity and modular
arithmetic features, prefix/suffix one-hot indicators, n-gram presence/counts,
run-length dynamics, and position-weighted features.
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Set
import numpy as np


# Basic-only: honest ML baseline with length, character counts/ratios, n-grams, and run dynamics.
# Strictly NO mod-k parity, NO prefix/suffix affixes, and NO oracle predicates.
BASIC_FEATURE_GROUPS = [
    "basic",
    "ngrams",
    "runs",
]

# Generic: comprehensive domain-agnostic regular language features (including mod-k and affixes)
GENERIC_FEATURE_GROUPS = [
    "basic",
    "parity",
    "affixes",
    "ngrams",
    "runs",
    "position_weighted",
]

ORACLE_FEATURE_GROUP = "oracle"

# Default to generic only (excluding leaky language-specific predicates)
FEATURE_GROUPS = GENERIC_FEATURE_GROUPS
ALL_FEATURE_GROUPS = GENERIC_FEATURE_GROUPS + [ORACLE_FEATURE_GROUP]

FEATURE_MODES: Dict[str, List[str]] = {
    "basic": BASIC_FEATURE_GROUPS,
    "generic": GENERIC_FEATURE_GROUPS,
    "oracle": ALL_FEATURE_GROUPS,
}


class StringFeatureExtractor:
    """Extracts numerical features from string sequences for scikit-learn models."""

    def __init__(
        self,
        alphabet: List[str],
        enabled_groups: Optional[List[str]] = None,
        max_ngram_n: int = 3,
        k_affix: int = 2,
    ) -> None:
        """Initialize feature extractor.

        Args:
            alphabet: Target alphabet characters (e.g. ['0', '1'] or ['a', 'b']).
            enabled_groups: List of feature groups to include. Defaults to all.
            max_ngram_n: Max n-gram length (1 to 3).
            k_affix: Max length for prefix and suffix features.
        """
        self.alphabet = sorted(list(alphabet))
        self.enabled_groups = set(enabled_groups if enabled_groups is not None else FEATURE_GROUPS)
        self.max_ngram_n = max_ngram_n
        self.k_affix = k_affix

        # Fitted vocabulary
        self.feature_names_: List[str] = []
        self.seen_prefixes_: Set[str] = set()
        self.seen_suffixes_: Set[str] = set()
        self.ngram_vocab_: List[str] = []
        self.is_fitted_ = False

    def fit(self, strings: List[str]) -> StringFeatureExtractor:
        """Fit extractor vocabulary on training strings."""
        self.seen_prefixes_ = set()
        self.seen_suffixes_ = set()
        observed_ngrams: Set[str] = set()

        for s in strings:
            for k in range(1, self.k_affix + 1):
                if len(s) >= k:
                    self.seen_prefixes_.add(s[:k])
                    self.seen_suffixes_.add(s[-k:])
            for n in range(1, self.max_ngram_n + 1):
                if len(s) >= n:
                    for i in range(len(s) - n + 1):
                        observed_ngrams.add(s[i:i + n])

        # Also add all single characters and pairs from alphabet if not present
        for c in self.alphabet:
            observed_ngrams.add(c)
            for c2 in self.alphabet:
                observed_ngrams.add(c + c2)

        self.ngram_vocab_ = sorted(list(observed_ngrams))
        self._build_feature_names()
        self.is_fitted_ = True
        return self

    def _build_feature_names(self) -> None:
        """Construct deterministic feature name list based on enabled groups."""
        names: List[str] = []

        if "basic" in self.enabled_groups:
            names.append("length")
            names.append("is_empty")
            for c in self.alphabet:
                names.append(f"count_{c}")
                names.append(f"ratio_{c}")

        if "parity" in self.enabled_groups:
            names.append("length_mod2")
            names.append("length_mod3")
            names.append("length_mod4")
            for c in self.alphabet:
                names.append(f"parity_{c}_mod2")
                names.append(f"parity_{c}_mod3")

        if "affixes" in self.enabled_groups:
            for pref in sorted(self.seen_prefixes_):
                names.append(f"prefix_{pref}")
            for suff in sorted(self.seen_suffixes_):
                names.append(f"suffix_{suff}")

        if "ngrams" in self.enabled_groups:
            for ng in self.ngram_vocab_:
                names.append(f"ngram_cnt_{ng}")
                names.append(f"ngram_has_{ng}")

        if "runs" in self.enabled_groups:
            for c in self.alphabet:
                names.append(f"max_run_{c}")
            names.append("num_alternations")
            names.append("num_runs")

        if "position_weighted" in self.enabled_groups:
            for c in self.alphabet:
                names.append(f"weighted_pos_{c}")
                names.append(f"alternating_sign_{c}")

        if "oracle" in self.enabled_groups:
            names.append("oracle_suffix_ab")
            names.append("oracle_even_0_odd_1")
            names.append("oracle_contains_101")
            names.append("oracle_count_a_div_3")
            names.append("oracle_binary_div_3")
            names.append("oracle_no_consecutive_1s")
            names.append("oracle_length_mod4_0")
            names.append("oracle_binary_mod3")

        self.feature_names_ = names

    def get_feature_names(self) -> List[str]:
        """Return list of extracted feature names."""
        if not self.is_fitted_:
            raise ValueError("Extractor is not fitted yet.")
        return self.feature_names_

    def _extract_single(self, s: str) -> List[float]:
        """Extract feature values for a single string."""
        vals: List[float] = []
        n_len = len(s)

        if "basic" in self.enabled_groups:
            vals.append(float(n_len))
            vals.append(1.0 if n_len == 0 else 0.0)
            for c in self.alphabet:
                cnt = s.count(c)
                vals.append(float(cnt))
                vals.append(float(cnt) / float(max(1, n_len)))

        if "parity" in self.enabled_groups:
            vals.append(float(n_len % 2))
            vals.append(float(n_len % 3))
            vals.append(float(n_len % 4))
            for c in self.alphabet:
                cnt = s.count(c)
                vals.append(float(cnt % 2))
                vals.append(float(cnt % 3))

        if "affixes" in self.enabled_groups:
            for pref in sorted(self.seen_prefixes_):
                vals.append(1.0 if s.startswith(pref) else 0.0)
            for suff in sorted(self.seen_suffixes_):
                vals.append(1.0 if s.endswith(suff) else 0.0)

        if "ngrams" in self.enabled_groups:
            for ng in self.ngram_vocab_:
                # Count occurrences of ng in s
                cnt = 0
                if len(ng) <= n_len:
                    for i in range(n_len - len(ng) + 1):
                        if s[i:i + len(ng)] == ng:
                            cnt += 1
                vals.append(float(cnt))
                vals.append(1.0 if cnt > 0 else 0.0)

        if "runs" in self.enabled_groups:
            for c in self.alphabet:
                # Max consecutive run of character c
                max_run = 0
                curr_run = 0
                for ch in s:
                    if ch == c:
                        curr_run += 1
                        if curr_run > max_run:
                            max_run = curr_run
                    else:
                        curr_run = 0
                vals.append(float(max_run))

            # Alternations and blocks
            if n_len <= 1:
                vals.append(0.0)
                vals.append(float(n_len))
            else:
                alternations = sum(1 for i in range(n_len - 1) if s[i] != s[i + 1])
                vals.append(float(alternations))
                vals.append(float(alternations + 1))

        if "position_weighted" in self.enabled_groups:
            denom = float(max(1, n_len * n_len))
            for c in self.alphabet:
                w_pos = sum(i + 1 for i, ch in enumerate(s) if ch == c)
                vals.append(float(w_pos) / denom)
                alt_sum = sum(((-1) ** i) for i, ch in enumerate(s) if ch == c)
                vals.append(float(alt_sum))

        if "oracle" in self.enabled_groups:
            is_bin = (n_len > 0 and set(s).issubset({"0", "1"}))
            vals.append(1.0 if s.endswith("ab") else 0.0)
            vals.append(1.0 if (s.count("0") % 2 == 0 and s.count("1") % 2 == 1) else 0.0)
            vals.append(1.0 if "101" in s else 0.0)
            vals.append(1.0 if (s.count("a") % 3 == 0) else 0.0)
            vals.append(1.0 if (is_bin and int(s, 2) % 3 == 0) else 0.0)
            vals.append(1.0 if "11" not in s else 0.0)
            vals.append(1.0 if (n_len % 4 == 0) else 0.0)
            vals.append(float(int(s, 2) % 3) if is_bin else 0.0)

        return vals

    def transform(self, strings: List[str]) -> np.ndarray:
        """Transform a list of strings into a 2D numpy feature matrix."""
        if not self.is_fitted_:
            raise ValueError("Extractor must be fitted before transform.")
        if not strings:
            return np.empty((0, len(self.feature_names_)), dtype=np.float32)

        data = [self._extract_single(s) for s in strings]
        return np.array(data, dtype=np.float32)

    def fit_transform(self, strings: List[str]) -> np.ndarray:
        """Fit extractor on strings and transform."""
        return self.fit(strings).transform(strings)
