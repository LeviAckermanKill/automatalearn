from __future__ import annotations

from typing import Any, Dict, Optional

from automatalearn.data.generator import DataGenerator


def ensure_seed(session_state: Dict[str, Any], fallback: int = 42) -> int:
    """Normalize the app's seed into both session keys used across the app."""
    raw_seed = session_state.get("global_seed", session_state.get("seed", fallback))
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        seed = int(fallback)
    session_state["global_seed"] = seed
    session_state["seed"] = seed
    return seed


def reset_language_cache(session_state: Dict[str, Any]) -> None:
    """Clear stale learned models whenever the language or dataset changes."""
    session_state["rpni_result"] = None
    session_state["lstar_result"] = None
    session_state["ml_results"] = {}
    session_state["benchmark_suite"] = None
    session_state["extractor"] = None
    session_state["classify_trained_lang"] = None


def rebuild_dataset_for_language(
    session_state: Dict[str, Any],
    language: Any,
    *,
    seed: Optional[int] = None,
    min_length: int = 0,
    max_length: int = 12,
    num_samples: int = 1500,
    gen_samples: int = 500,
    class_balance: float = 0.5,
    noise_rate: float = 0.0,
) -> Any:
    final_seed = ensure_seed(session_state, fallback=seed if seed is not None else 42)
    gen = DataGenerator(
        alphabet=language.alphabet,
        oracle=language.predicate,
        dfa=language.dfa,
        seed=final_seed,
    )
    session_state["splits"] = gen.generate_dataset(
        min_length=min_length,
        max_length=max_length,
        num_samples=num_samples,
        class_balance=class_balance,
        noise_rate=noise_rate,
        gen_min_length=max_length + 1,
        gen_max_length=max_length + 12,
        gen_samples=gen_samples,
    )
    reset_language_cache(session_state)
    return session_state["splits"]


__all__ = [
    "ensure_seed",
    "reset_language_cache",
    "rebuild_dataset_for_language",
]
