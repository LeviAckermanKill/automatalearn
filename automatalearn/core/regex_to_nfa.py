"""Thompson's Construction: Regular Expression to NFA compiler.

Converts standard regular expressions with union (|), concatenation,
Kleene star (*), plus (+), optional (?), and grouping parentheses into
an equivalent NFA with epsilon transitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from automatalearn.core.dfa import State
from automatalearn.core.nfa import NFA


@dataclass
class _Fragment:
    """Internal helper representing a sub-NFA with single start and accept state."""
    start: int
    accept: int
    transitions: Dict[Tuple[int, str], Set[int]]
    alphabet: Set[str]


class RegexParser:
    """Parser and Thompson compiler for regular expressions."""

    def __init__(self) -> None:
        self._state_counter = 0

    def _new_state(self) -> int:
        st = self._state_counter
        self._state_counter += 1
        return st

    def _insert_explicit_concat(self, regex: str) -> str:
        """Insert explicit concatenation '.' operators into regex."""
        if not regex:
            return ""
        output: List[str] = []
        operators = {"|", "*", "+", "?", "(", ")"}

        for i, char in enumerate(regex):
            output.append(char)
            if i + 1 < len(regex):
                next_char = regex[i + 1]
                # Can char be followed by concatenation?
                # char can be a literal, or *, +, ?, or )
                left_operand = (char not in operators) or (char in {"*", "+", "?", ")"})
                # next_char can be a literal or (
                right_operand = (next_char not in operators) or (next_char == "(")

                if left_operand and right_operand:
                    output.append(".")

        return "".join(output)

    def _to_postfix(self, regex: str) -> str:
        """Convert regex with explicit concatenation to postfix via Shunting-Yard."""
        precedence = {"*": 3, "+": 3, "?": 3, ".": 2, "|": 1}
        output: List[str] = []
        op_stack: List[str] = []

        for char in regex:
            if char in {"*", "+", "?"}:
                output.append(char)
            elif char in {".", "|"}:
                while op_stack and op_stack[-1] != "(" and precedence.get(op_stack[-1], 0) >= precedence[char]:
                    output.append(op_stack.pop())
                op_stack.append(char)
            elif char == "(":
                op_stack.append(char)
            elif char == ")":
                while op_stack and op_stack[-1] != "(":
                    output.append(op_stack.pop())
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
            else:
                output.append(char)

        while op_stack:
            output.append(op_stack.pop())

        return "".join(output)

    def compile(self, regex_str: str) -> NFA:
        """Compile a regex string into an NFA.

        Args:
            regex_str: Regular expression string (e.g., '(a|b)*ab').

        Returns:
            Equivalent NFA.
        """
        self._state_counter = 0
        cleaned = regex_str.strip().replace(" ", "")

        if not cleaned:
            # Empty string automaton
            s = self._new_state()
            return NFA(states={s}, alphabet=set(), transitions={}, start_states={s}, accept_states={s})

        explicit = self._insert_explicit_concat(cleaned)
        postfix = self._to_postfix(explicit)

        stack: List[_Fragment] = []

        for token in postfix:
            if token == ".":
                # Concatenation: A . B
                if len(stack) < 2:
                    continue
                frag_b = stack.pop()
                frag_a = stack.pop()
                trans = dict(frag_a.transitions)
                for k, v in frag_b.transitions.items():
                    trans.setdefault(k, set()).update(v)
                # Connect A's accept to B's start with epsilon
                trans.setdefault((frag_a.accept, ""), set()).add(frag_b.start)
                stack.append(_Fragment(
                    start=frag_a.start,
                    accept=frag_b.accept,
                    transitions=trans,
                    alphabet=frag_a.alphabet | frag_b.alphabet,
                ))

            elif token == "|":
                # Union: A | B
                if len(stack) < 2:
                    continue
                frag_b = stack.pop()
                frag_a = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag_a.transitions)
                for k, v in frag_b.transitions.items():
                    trans.setdefault(k, set()).update(v)

                trans.setdefault((s_start, ""), set()).update({frag_a.start, frag_b.start})
                trans.setdefault((frag_a.accept, ""), set()).add(s_accept)
                trans.setdefault((frag_b.accept, ""), set()).add(s_accept)

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag_a.alphabet | frag_b.alphabet,
                ))

            elif token == "*":
                # Kleene Star: A*
                if not stack:
                    continue
                frag = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag.transitions)

                trans.setdefault((s_start, ""), set()).update({frag.start, s_accept})
                trans.setdefault((frag.accept, ""), set()).update({frag.start, s_accept})

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag.alphabet,
                ))

            elif token == "+":
                # One or more: A+ = A . A*
                if not stack:
                    continue
                frag = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag.transitions)

                trans.setdefault((s_start, ""), set()).add(frag.start)
                trans.setdefault((frag.accept, ""), set()).update({frag.start, s_accept})

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag.alphabet,
                ))

            elif token == "?":
                # Zero or one: A?
                if not stack:
                    continue
                frag = stack.pop()
                s_start = self._new_state()
                s_accept = self._new_state()
                trans = dict(frag.transitions)

                trans.setdefault((s_start, ""), set()).update({frag.start, s_accept})
                trans.setdefault((frag.accept, ""), set()).add(s_accept)

                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=frag.alphabet,
                ))

            else:
                # Literal character
                s_start = self._new_state()
                s_accept = self._new_state()
                char = "" if token in {"ε", "\\e"} else token
                alpha = set() if char == "" else {char}
                trans = {(s_start, char): {s_accept}}
                stack.append(_Fragment(
                    start=s_start,
                    accept=s_accept,
                    transitions=trans,
                    alphabet=alpha,
                ))

        if not stack:
            s = self._new_state()
            return NFA(states={s}, alphabet=set(), transitions={}, start_states={s}, accept_states={s})

        root = stack.pop()
        # Collect all states
        all_states: Set[State] = {root.start, root.accept}
        for (st, _), dsts in root.transitions.items():
            all_states.add(st)
            all_states.update(dsts)

        return NFA(
            states=all_states,
            alphabet=root.alphabet,
            transitions=root.transitions,
            start_states={root.start},
            accept_states={root.accept},
            name=f"NFA_regex_{regex_str}",
        )


def regex_to_nfa(regex_str: str) -> NFA:
    """Convenience function compiling regex to NFA."""
    parser = RegexParser()
    return parser.compile(regex_str)
