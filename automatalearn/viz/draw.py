"""Graphviz drawing and DOT generation for DFAs, NFAs, and Trees.

Formats clean, modern dark-mode automata diagrams (Linear / W&B style) with
transparent background, neon emerald accepting states, vibrant active path highlighting,
escaped string labels, and dead-state red highlighting for sink/unspecified transitions.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import graphviz
from automatalearn.core.dfa import DFA, State


def _escape_dot(s: Any) -> str:
    """Safely escape strings for Graphviz DOT format to prevent syntax errors."""
    text = str(s)
    return text.replace('\\', '\\\\').replace('"', '\\"')


def _is_dead_state(dfa: DFA, state: State) -> bool:
    """Heuristic: a dead state has no transitions for any symbol in the alphabet."""
    return all(dfa.transitions.get((state, symbol)) is None for symbol in sorted(dfa.alphabet))


def dfa_to_dot(
    dfa: DFA,
    highlight_states: Optional[Set[State]] = None,
    active_path: Optional[List[State]] = None,
    title: Optional[str] = None,
) -> str:
    """Generate a clean dark-mode DOT representation of a DFA with escaped labels."""
    lines: List[str] = []
    lines.append('digraph DFA {')
    lines.append('  rankdir=LR;')
    lines.append('  bgcolor="transparent";')
    lines.append('  size="6,2.6";')
    lines.append('  ratio=compress;')
    lines.append('  nodesep=0.4;')
    lines.append('  ranksep=0.6;')
    lines.append('  node [fontname="Inter, -apple-system, sans-serif", fontsize=11, height=0.45, width=0.45];')
    lines.append('  edge [fontname="JetBrains Mono, monospace", fontsize=10];')

    esc_start = _escape_dot(dfa.start_state)
    lines.append('  __start__ [shape=none, label="", width=0, height=0];')
    lines.append(f'  __start__ -> "{esc_start}" [color="#22C55E", penwidth=2.0];')

    highlights = set(highlight_states or [])
    current_active = active_path[-1] if active_path else None

    for st in sorted(dfa.states, key=str):
        esc_st = _escape_dot(st)
        is_accept = st in dfa.accept_states
        is_highlight = st in highlights
        is_active = (st == current_active)
        is_dead = _is_dead_state(dfa, st)

        shape = "doublecircle" if is_accept else "circle"

        if st == dfa.start_state:
            fillcolor = "#064E3B"
            bordercolor = "#34D399"
            fontcolor = "#ECFDF5"
            penwidth = "2.4"
        elif is_active:
            fillcolor = "#4F46E5"
            bordercolor = "#A5B4FC"
            fontcolor = "#FFFFFF"
            penwidth = "2.6"
        elif is_highlight:
            fillcolor = "#1E3A8A"
            bordercolor = "#60A5FA"
            fontcolor = "#EFF6FF"
            penwidth = "2.0"
        elif is_dead:
            fillcolor = "#7F1D1D"
            bordercolor = "#F87171"
            fontcolor = "#FEE2E2"
            penwidth = "2.0"
        elif is_accept:
            fillcolor = "#064E3B"
            bordercolor = "#10B981"
            fontcolor = "#ECFDF5"
            penwidth = "2.2"
        else:
            fillcolor = "#111827"
            bordercolor = "#374151"
            fontcolor = "#F3F4F6"
            penwidth = "1.5"

        lines.append(
            f'  "{esc_st}" [shape={shape}, style="filled", fillcolor="{fillcolor}", '
            f'color="{bordercolor}", fontcolor="{fontcolor}", penwidth={penwidth}];'
        )

    edge_map: Dict[Tuple[State, State], List[str]] = {}
    for (u, sym), v in sorted(dfa.transitions.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        edge_map.setdefault((u, v), []).append(sym)

    for (u, v), syms in sorted(edge_map.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        esc_u = _escape_dot(u)
        esc_v = _escape_dot(v)
        label = _escape_dot(", ".join(sorted(syms)))

        edge_style = 'color="#4B5563", fontcolor="#D1D5DB", penwidth=1.2'
        if active_path and len(active_path) >= 2:
            last_u, last_v = active_path[-2], active_path[-1]
            if last_u == u and last_v == v:
                edge_style = 'color="#6C8CFF", fontcolor="#A5B4FC", penwidth=2.5'

        lines.append(f'  "{esc_u}" -> "{esc_v}" [label=" {label} ", {edge_style}];')

    lines.append('}')
    return "\n".join(lines)


def dfa_to_graphviz(
    dfa: DFA,
    highlight_states: Optional[Set[State]] = None,
    active_path: Optional[List[State]] = None,
    title: Optional[str] = None,
) -> graphviz.Digraph:
    """Generate a graphviz.Digraph object from a DFA."""
    dot_str = dfa_to_dot(dfa, highlight_states, active_path, title)
    return graphviz.Source(dot_str)


def nfa_to_dot(
    nfa: Any,
    highlight_states: Optional[Set[State]] = None,
    title: Optional[str] = None,
) -> str:
    """Generate clean dark-mode DOT string representation of an NFA with escaped labels."""
    lines: List[str] = []
    lines.append('digraph NFA {')
    lines.append('  rankdir=LR;')
    lines.append('  bgcolor="transparent";')
    lines.append('  size="6,2.6";')
    lines.append('  ratio=compress;')
    lines.append('  nodesep=0.4;')
    lines.append('  ranksep=0.6;')
    lines.append('  node [fontname="Inter, -apple-system, sans-serif", fontsize=11, height=0.45, width=0.45];')
    lines.append('  edge [fontname="JetBrains Mono, monospace", fontsize=10];')

    for idx, s in enumerate(sorted(nfa.start_states, key=str)):
        esc_s = _escape_dot(s)
        lines.append(f'  __start_{idx}__ [shape=none, label="", width=0, height=0];')
        lines.append(f'  __start_{idx}__ -> "{esc_s}" [color="#22C55E", penwidth=2.0];')

    highlights = set(highlight_states or [])

    for st in sorted(nfa.states, key=str):
        esc_st = _escape_dot(st)
        is_accept = st in nfa.accept_states
        is_highlight = st in highlights

        shape = "doublecircle" if is_accept else "circle"

        if is_highlight:
            fillcolor = "#1E3A8A"
            bordercolor = "#60A5FA"
            fontcolor = "#EFF6FF"
            penwidth = "2.0"
        elif is_accept:
            fillcolor = "#064E3B"
            bordercolor = "#10B981"
            fontcolor = "#ECFDF5"
            penwidth = "2.2"
        else:
            fillcolor = "#111827"
            bordercolor = "#374151"
            fontcolor = "#F3F4F6"
            penwidth = "1.5"

        lines.append(
            f'  "{esc_st}" [shape={shape}, style="filled", fillcolor="{fillcolor}", '
            f'color="{bordercolor}", fontcolor="{fontcolor}", penwidth={penwidth}];'
        )

    edge_map: Dict[Tuple[State, State], List[str]] = {}
    for (u, sym), dsts in nfa.transitions.items():
        disp_sym = "ε" if sym == "" else sym
        for v in dsts:
            edge_map.setdefault((u, v), []).append(disp_sym)

    for (u, v), syms in sorted(edge_map.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        esc_u = _escape_dot(u)
        esc_v = _escape_dot(v)
        label = _escape_dot(", ".join(sorted(syms)))
        lines.append(f'  "{esc_u}" -> "{esc_v}" [label=" {label} ", color="#4B5563", fontcolor="#D1D5DB", penwidth=1.2];')

    lines.append('}')
    return "\n".join(lines)


def nfa_to_graphviz(
    nfa: Any,
    highlight_states: Optional[Set[State]] = None,
    title: Optional[str] = None,
) -> graphviz.Digraph:
    """Generate graphviz.Digraph representation of an NFA."""
    dot_str = nfa_to_dot(nfa, highlight_states, title)
    return graphviz.Source(dot_str)
