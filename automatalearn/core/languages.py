"""Predefined benchmark regular languages and their ground-truth DFAs.

Provides 7 canonical regular languages representing diverse computational
challenges (modular arithmetic, parity tracking, local substring patterns,
and suffix matching) along with ground-truth predicates and DFAs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from automatalearn.core.dfa import DFA


@dataclass
class LanguageDefinition:
    """Specification of a benchmark regular language."""

    key: str
    name: str
    description: str
    alphabet: List[str]
    dfa: DFA
    predicate: Callable[[str], bool]
    regex: Optional[str] = None
    theoretical_notes: str = ""

    def accepts(self, s: str) -> bool:
        """Ground-truth oracle test."""
        return self.predicate(s)


def build_ending_with_ab() -> LanguageDefinition:
    """Language: Strings over {a, b} ending with 'ab'."""
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
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q2"},
        name="EndingWith_ab",
    )
    return LanguageDefinition(
        key="ending_ab",
        name='Strings ending with "ab"',
        description="Alphabet {a, b}. Accepts strings that terminate with suffix 'ab'.",
        alphabet=["a", "b"],
        dfa=dfa,
        predicate=lambda s: s.endswith("ab"),
        regex=r"(a|b)*ab",
        theoretical_notes=(
            "Requires tracking the last two symbols. Decision trees with suffix features "
            "perform well, but without position/suffix features classical ML must learn "
            "n-gram positions."
        ),
    )


def build_even_0_odd_1() -> LanguageDefinition:
    """Language: Even number of 0s AND odd number of 1s over {0, 1}."""
    states = {"q_EE", "q_EO", "q_OE", "q_OO"}
    alphabet = {"0", "1"}
    transitions = {
        ("q_EE", "0"): "q_OE",
        ("q_EE", "1"): "q_EO",
        ("q_EO", "0"): "q_OO",
        ("q_EO", "1"): "q_EE",
        ("q_OE", "0"): "q_EE",
        ("q_OE", "1"): "q_OO",
        ("q_OO", "0"): "q_EO",
        ("q_OO", "1"): "q_OE",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q_EE",
        accept_states={"q_EO"},
        name="Even0_Odd1",
    )
    return LanguageDefinition(
        key="even_0_odd_1",
        name="Even number of 0s AND odd number of 1s",
        description="Alphabet {0, 1}. State tracks parity of both symbols simultaneously (Z_2 x Z_2).",
        alphabet=["0", "1"],
        dfa=dfa,
        predicate=lambda s: (s.count("0") % 2 == 0) and (s.count("1") % 2 == 1),
        regex=None,
        theoretical_notes=(
            "Classic parity problem (XOR-like). Highly non-linear for linear models. "
            "Decision trees struggle without explicit parity mod 2 features."
        ),
    )


def build_contains_101() -> LanguageDefinition:
    """Language: Contains substring '101' over {0, 1}."""
    states = {"q0", "q1", "q2", "q3"}
    alphabet = {"0", "1"}
    transitions = {
        ("q0", "0"): "q0",
        ("q0", "1"): "q1",
        ("q1", "0"): "q2",
        ("q1", "1"): "q1",
        ("q2", "0"): "q0",
        ("q2", "1"): "q3",
        ("q3", "0"): "q3",
        ("q3", "1"): "q3",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q3"},
        name="Contains_101",
    )
    return LanguageDefinition(
        key="contains_101",
        name='Contains substring "101"',
        description="Alphabet {0, 1}. Accepts if '101' appears anywhere as a contiguous substring.",
        alphabet=["0", "1"],
        dfa=dfa,
        predicate=lambda s: "101" in s,
        regex=r"(0|1)*101(0|1)*",
        theoretical_notes=(
            "Local pattern matching with an absorbing accept state (q3). "
            "ML with 3-gram feature '101' achieves high accuracy, but fails if length increases "
            "and feature vocabulary is restricted."
        ),
    )


def build_count_a_div_3() -> LanguageDefinition:
    """Language: Number of a's is divisible by 3 over {a, b}."""
    states = {"q0", "q1", "q2"}
    alphabet = {"a", "b"}
    transitions = {
        ("q0", "a"): "q1",
        ("q0", "b"): "q0",
        ("q1", "a"): "q2",
        ("q1", "b"): "q1",
        ("q2", "a"): "q0",
        ("q2", "b"): "q2",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q0"},
        name="CountA_Div3",
    )
    return LanguageDefinition(
        key="count_a_div_3",
        name="Number of a's divisible by 3",
        description="Alphabet {a, b}. Accepts if count('a') mod 3 == 0 (including 0).",
        alphabet=["a", "b"],
        dfa=dfa,
        predicate=lambda s: s.count("a") % 3 == 0,
        regex=r"(b*ab*ab*a)*b*",
        theoretical_notes=(
            "Modulo counting language (Z_3). Linear classifiers cannot compute modulo arithmetic "
            "without explicit mod 3 features. DFA represents this with exactly 3 states."
        ),
    )


def build_binary_div_3() -> LanguageDefinition:
    """Language: Binary numbers divisible by 3 over {0, 1}."""
    states = {"q0", "q1", "q2"}
    alphabet = {"0", "1"}
    transitions = {
        ("q0", "0"): "q0",
        ("q0", "1"): "q1",
        ("q1", "0"): "q2",
        ("q1", "1"): "q0",
        ("q2", "0"): "q1",
        ("q2", "1"): "q2",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q0"},
        name="Binary_Div3",
    )

    def _div3_predicate(s: str) -> bool:
        if not s:
            return True
        val = int(s, 2)
        return val % 3 == 0

    return LanguageDefinition(
        key="binary_div_3",
        name="Binary numbers divisible by 3",
        description="Alphabet {0, 1}. Interprets string as big-endian binary integer mod 3.",
        alphabet=["0", "1"],
        dfa=dfa,
        predicate=_div3_predicate,
        regex=None,
        theoretical_notes=(
            "Classic Horner's method automaton: r' = (2*r + b) mod 3. "
            "ML models fail dramatically on longer strings because position value alternates "
            "powers of 2 mod 3 (1, 2, 1, 2, ...)."
        ),
    )


def build_no_consecutive_1s() -> LanguageDefinition:
    """Language: No two consecutive 1s over {0, 1}."""
    states = {"q0", "q1", "q_dead"}
    alphabet = {"0", "1"}
    transitions = {
        ("q0", "0"): "q0",
        ("q0", "1"): "q1",
        ("q1", "0"): "q0",
        ("q1", "1"): "q_dead",
        ("q_dead", "0"): "q_dead",
        ("q_dead", "1"): "q_dead",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q0", "q1"},
        name="NoConsecutive1s",
    )
    return LanguageDefinition(
        key="no_consecutive_1s",
        name="No two consecutive 1s",
        description="Alphabet {0, 1}. Strings containing no occurrence of '11'.",
        alphabet=["0", "1"],
        dfa=dfa,
        predicate=lambda s: "11" not in s,
        regex=r"(0|10)*(1|ε)",
        theoretical_notes=(
            "Fibonacci language: the number of accepted strings of length n is Fib(n+2). "
            "Contains an absorbing dead state q_dead."
        ),
    )


def build_length_mod_4() -> LanguageDefinition:
    """Language: Length mod 4 == 0 over {0, 1}."""
    states = {"q0", "q1", "q2", "q3"}
    alphabet = {"0", "1"}
    transitions = {
        ("q0", "0"): "q1",
        ("q0", "1"): "q1",
        ("q1", "0"): "q2",
        ("q1", "1"): "q2",
        ("q2", "0"): "q3",
        ("q2", "1"): "q3",
        ("q3", "0"): "q0",
        ("q3", "1"): "q0",
    }
    dfa = DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q0"},
        name="Length_Mod4",
    )
    return LanguageDefinition(
        key="length_mod_4",
        name="Length mod 4 == 0",
        description="Alphabet {0, 1}. Accepts if len(s) % 4 == 0 (including empty string).",
        alphabet=["0", "1"],
        dfa=dfa,
        predicate=lambda s: len(s) % 4 == 0,
        regex=r"((0|1){4})*",
        theoretical_notes=(
            "Depends entirely on length, ignoring symbol identities. "
            "Highlights whether models extrapolate cyclic periodicity to unseen long strings."
        ),
    )


PREDEFINED_LANGUAGES: Dict[str, Callable[[], LanguageDefinition]] = {
    "ending_ab": build_ending_with_ab,
    "even_0_odd_1": build_even_0_odd_1,
    "contains_101": build_contains_101,
    "count_a_div_3": build_count_a_div_3,
    "binary_div_3": build_binary_div_3,
    "no_consecutive_1s": build_no_consecutive_1s,
    "length_mod_4": build_length_mod_4,
}


def get_language(key: str) -> LanguageDefinition:
    """Retrieve language definition by key."""
    if key not in PREDEFINED_LANGUAGES:
        raise KeyError(f"Unknown language key: {key}. Available: {list(PREDEFINED_LANGUAGES.keys())}")
    return PREDEFINED_LANGUAGES[key]()


def list_available_languages() -> List[Dict[str, str]]:
    """Return summary metadata for all predefined languages."""
    result = []
    for k, builder in PREDEFINED_LANGUAGES.items():
        lang = builder()
        result.append({
            "key": k,
            "name": lang.name,
            "description": lang.description,
            "alphabet": ", ".join(lang.alphabet),
            "states": str(len(lang.dfa.states)),
        })
    return result
