"""Dataset generation for regular language learning and classification.

Supports uniform random generation, exhaustive short-string generation,
targeted DFA random walks for balanced sampling, label noise injection,
train/validation/test splitting, and dedicated out-of-distribution
length-generalization test sets.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from automatalearn.core.dfa import DFA


@dataclass
class DatasetSplits:
    """Container for partitioned datasets."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    generalization_df: pd.DataFrame

    @property
    def train_strings(self) -> List[str]:
        return self.train_df["string"].tolist()

    @property
    def train_labels(self) -> List[int]:
        return self.train_df["label"].tolist()

    @property
    def test_strings(self) -> List[str]:
        return self.test_df["string"].tolist()

    @property
    def test_labels(self) -> List[int]:
        return self.test_df["label"].tolist()

    @property
    def gen_strings(self) -> List[str]:
        return self.generalization_df["string"].tolist()

    @property
    def gen_labels(self) -> List[int]:
        return self.generalization_df["label"].tolist()

    @property
    def alphabet(self) -> List[str]:
        chars = set()
        for df in (self.train_df, self.val_df, self.test_df, self.generalization_df):
            if df is not None and "string" in df.columns:
                for s in df["string"]:
                    chars.update(s)
        return sorted(list(chars))


class DataGenerator:
    """Generates synthetic datasets for automata and regular language experiments."""

    def __init__(
        self,
        alphabet: List[str],
        oracle: Callable[[str], bool],
        dfa: Optional[DFA] = None,
        seed: int = 42,
    ) -> None:
        """Initialize generator.

        Args:
            alphabet: List of valid alphabet characters (e.g. ['0', '1']).
            oracle: Function mapping string -> bool (ground truth).
            dfa: Optional ground-truth DFA for targeted random walk sampling.
            seed: Random seed for deterministic reproducibility.
        """
        self.alphabet = sorted(list(alphabet))
        self.oracle = oracle
        self.dfa = dfa
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_random_walk_string(self, target_accept: bool, max_len: int) -> Optional[str]:
        """Generate a string matching target_accept by random walk on the DFA."""
        if not self.dfa:
            return None
        # Attempt random walks up to length max_len
        for _ in range(50):
            curr = self.dfa.start_state
            chars = []
            walk_len = self.rng.randint(0, max_len)
            for _ in range(walk_len):
                c = self.rng.choice(self.alphabet)
                nxt = self.dfa.transition(curr, c)
                if nxt is None:
                    break
                curr = nxt
                chars.append(c)
            s = "".join(chars)
            if self.oracle(s) == target_accept:
                return s
        return None

    def generate_string_pool(
        self,
        min_length: int,
        max_length: int,
        num_samples: int,
        class_balance: float = 0.5,
    ) -> List[Tuple[str, int]]:
        """Generate a balanced pool of unique labeled strings.

        Args:
            min_length: Minimum string length (>= 0).
            max_length: Maximum string length (>= min_length).
            num_samples: Target number of unique samples.
            class_balance: Target proportion of positive samples (0.0 to 1.0).

        Returns:
            List of (string, label) tuples.
        """
        target_pos = int(round(num_samples * class_balance))
        target_neg = num_samples - target_pos

        pos_set: Set[str] = set()
        neg_set: Set[str] = set()

        # Phase 1: Exhaustive enumeration for very short lengths
        exhaustive_limit = min(5, max_length)
        queue = [""]
        while queue:
            curr = queue.pop(0)
            if len(curr) >= min_length:
                is_pos = self.oracle(curr)
                if is_pos and len(pos_set) < target_pos:
                    pos_set.add(curr)
                elif not is_pos and len(neg_set) < target_neg:
                    neg_set.add(curr)

            if len(curr) < exhaustive_limit:
                for c in self.alphabet:
                    queue.append(curr + c)

        # Phase 2: Uniform random sampling
        max_attempts = num_samples * 150
        attempts = 0

        while (len(pos_set) < target_pos or len(neg_set) < target_neg) and attempts < max_attempts:
            attempts += 1
            length = self.rng.randint(min_length, max_length)
            candidate = "".join(self.rng.choices(self.alphabet, k=length))
            is_pos = self.oracle(candidate)

            if is_pos and len(pos_set) < target_pos:
                pos_set.add(candidate)
            elif not is_pos and len(neg_set) < target_neg:
                neg_set.add(candidate)

        # Phase 3: Targeted DFA walks if positive or negative class is still starved
        if len(pos_set) < target_pos and self.dfa is not None:
            while len(pos_set) < target_pos and attempts < max_attempts + 500:
                attempts += 1
                pos_walk = self.generate_random_walk_string(True, max_length)
                if pos_walk and len(pos_walk) >= min_length:
                    pos_set.add(pos_walk)

        if len(neg_set) < target_neg and self.dfa is not None:
            while len(neg_set) < target_neg and attempts < max_attempts + 500:
                attempts += 1
                neg_walk = self.generate_random_walk_string(False, max_length)
                if neg_walk and len(neg_walk) >= min_length:
                    neg_set.add(neg_walk)

        samples: List[Tuple[str, int]] = [(s, 1) for s in pos_set] + [(s, 0) for s in neg_set]
        self.rng.shuffle(samples)
        return samples

    def inject_noise(self, samples: List[Tuple[str, int]], noise_rate: float) -> List[Tuple[str, int]]:
        """Inject label-flipping noise into training samples.

        Args:
            samples: List of (string, label).
            noise_rate: Probability of flipping each label (0.0 to 0.5).

        Returns:
            Noisy samples with flipped labels.
        """
        if noise_rate <= 0.0:
            return samples

        noisy = []
        for s, y in samples:
            if self.rng.random() < noise_rate:
                noisy.append((s, 1 - y))
            else:
                noisy.append((s, y))
        return noisy

    def generate_dataset(
        self,
        min_length: int = 0,
        max_length: int = 12,
        num_samples: int = 1500,
        class_balance: float = 0.5,
        noise_rate: float = 0.0,
        train_val_test_ratio: Tuple[float, float, float] = (0.5, 0.15, 0.35),
        gen_min_length: Optional[int] = None,
        gen_max_length: Optional[int] = None,
        gen_samples: int = 500,
    ) -> DatasetSplits:
        """Generate full partitioned dataset with training, validation, test, and generalization sets.

        Args:
            min_length: Min length for in-distribution data.
            max_length: Max length for in-distribution data.
            num_samples: Total in-distribution sample count.
            class_balance: Proportion of positive samples.
            noise_rate: Label noise to apply to train set only.
            train_val_test_ratio: Split ratios for (train, val, test).
            gen_min_length: Min length for length-generalization set (defaults to max_length + 1).
            gen_max_length: Max length for length-generalization set (defaults to max_length + 10).
            gen_samples: Number of samples in generalization set.

        Returns:
            DatasetSplits containing train, val, test, and generalization DataFrames.
        """
        samples = self.generate_string_pool(
            min_length=min_length,
            max_length=max_length,
            num_samples=num_samples,
            class_balance=class_balance,
        )

        n_total = len(samples)
        r_train, r_val, _ = train_val_test_ratio
        n_train = int(round(n_total * r_train))
        n_val = int(round(n_total * r_val))

        train_pool = samples[:n_train]
        val_pool = samples[n_train:n_train + n_val]
        test_pool = samples[n_train + n_val:]

        # Apply noise ONLY to train set (validation and test remain uncorrupted ground truth)
        if noise_rate > 0.0:
            train_pool = self.inject_noise(train_pool, noise_rate)

        # Build length-generalization set (strictly longer strings)
        g_min = gen_min_length if gen_min_length is not None else max_length + 1
        g_max = gen_max_length if gen_max_length is not None else max_length + 10

        gen_pool = self.generate_string_pool(
            min_length=g_min,
            max_length=g_max,
            num_samples=gen_samples,
            class_balance=class_balance,
        )

        def make_df(data: List[Tuple[str, int]]) -> pd.DataFrame:
            if not data:
                return pd.DataFrame(columns=["string", "label", "length"])
            df = pd.DataFrame(data, columns=["string", "label"])
            df["length"] = df["string"].str.len()
            return df

        return DatasetSplits(
            train_df=make_df(train_pool),
            val_df=make_df(val_pool),
            test_df=make_df(test_pool),
            generalization_df=make_df(gen_pool),
        )


def load_dataset_from_csv(csv_content_or_path: str) -> pd.DataFrame:
    """Load and validate dataset from CSV string or filepath."""
    import io
    if "\n" in csv_content_or_path:
        df = pd.read_csv(io.StringIO(csv_content_or_path), dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(csv_content_or_path, dtype=str, keep_default_na=False)

    # Ensure required columns
    col_map = {c.lower().strip(): c for c in df.columns}
    if "string" not in col_map or "label" not in col_map:
        raise ValueError("CSV must contain 'string' and 'label' columns.")

    df = df.rename(columns={col_map["string"]: "string", col_map["label"]: "label"})
    df["string"] = df["string"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["length"] = df["string"].str.len()
    return df[["string", "label", "length"]]


def export_dataset_to_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string."""
    return df[["string", "label"]].to_csv(index=False)
