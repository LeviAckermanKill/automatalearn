"""Core automata module."""

from automatalearn.core.dfa import DFA
from automatalearn.core.nfa import NFA
from automatalearn.core.minimize import minimize_dfa, minimize_dfa_hopcroft, minimize_dfa_table_filling, MinimizationResult
from automatalearn.core.languages import get_language, list_available_languages, LanguageDefinition
from automatalearn.core.regex_to_nfa import regex_to_nfa

__all__ = [
    "DFA",
    "NFA",
    "minimize_dfa",
    "minimize_dfa_hopcroft",
    "minimize_dfa_table_filling",
    "MinimizationResult",
    "get_language",
    "list_available_languages",
    "LanguageDefinition",
    "regex_to_nfa",
]
