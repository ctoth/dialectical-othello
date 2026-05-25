"""Slow no-pruning reference minimax for differential testing chunk 2.

This intentionally implements the *same* evaluation contract as
``dialectical_othello.search.alphabeta`` but **without** alphabeta
pruning — every child of every internal node is visited and the
``max`` across negated child scores is returned in full.

The differential test (``tests/test_search_properties.py``) compares
the two at small depths: alphabeta with pruning must return the same
score as this slow reference at every shared depth. Any disagreement
is a soundness bug in the pruning logic.

Deliberately minimal: no deadline, no caching, no transposition table.
Used only by tests.
"""

from __future__ import annotations

from dialectical_othello.board import OthelloBoard
from dialectical_othello.search import TERMINAL_SCORE, static_evaluation


def _terminal_value(board: OthelloBoard) -> int:
    diff = static_evaluation(board)
    if diff > 0:
        return TERMINAL_SCORE + diff
    if diff < 0:
        return -TERMINAL_SCORE + diff
    return 0


def minimax(board: OthelloBoard, depth: int) -> int:
    """Return the mover-relative minimax score (no pruning)."""
    if board.is_terminal():
        return _terminal_value(board)
    if depth <= 0:
        return static_evaluation(board)
    moves = board.legal_moves()
    if not moves:
        return _terminal_value(board)
    best = -(TERMINAL_SCORE * 100)
    for move in moves:
        child = board.apply(move)
        score = -minimax(child, depth - 1)
        if score > best:
            best = score
    return best
