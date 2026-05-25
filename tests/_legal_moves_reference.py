"""Slow reference implementation of Othello legal-move enumeration.

This module exists ONLY for cross-checking
:meth:`dialectical_othello.board.OthelloBoard.legal_moves`. It is the
textbook 8-direction walk over each empty square: for each empty square
``s`` and each of the 8 directions ``d``, walk one square at a time from
``s`` in direction ``d``; if the first square holds an opponent disc and a
contiguous run of opponent discs ends with a mover disc, ``s`` is a legal
move for the mover.

Deliberately slow and obvious. No bitboards, no precomputed tables — just
direct (file, rank) arithmetic. Used by ``test_board_properties.py``.
"""

from __future__ import annotations

from dialectical_othello.board import NUM_SQUARES, OthelloBoard, OthelloMove

_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
)


def _cell(board: OthelloBoard, file: int, rank: int) -> str:
    """Return ``'B'``, ``'W'``, or ``'.'`` for the cell at (file, rank)."""
    if not (0 <= file < 8 and 0 <= rank < 8):
        return "X"  # off-board sentinel
    bit = 1 << (file + 8 * rank)
    if board.black & bit:
        return "B"
    if board.white & bit:
        return "W"
    return "."


def reference_legal_squares(board: OthelloBoard) -> set[int]:
    """Return the set of 0..63 squares where the side to move may play.

    Pure direction-walk. Returns the empty set if no disc-flip placement
    exists. Does NOT inject a pass move — callers compare against the
    bitboard ``legal_moves()`` result filtered to placement squares.
    """
    mover = board.turn
    opp = "W" if mover == "B" else "B"
    legal: set[int] = set()
    for sq in range(NUM_SQUARES):
        file = sq & 7
        rank = sq >> 3
        if _cell(board, file, rank) != ".":
            continue
        for dfile, drank in _DIRECTIONS:
            f, r = file + dfile, rank + drank
            if _cell(board, f, r) != opp:
                continue
            # Walk further opponent discs.
            f += dfile
            r += drank
            while _cell(board, f, r) == opp:
                f += dfile
                r += drank
            if _cell(board, f, r) == mover:
                legal.add(sq)
                break
    return legal


def reference_legal_moves(board: OthelloBoard) -> tuple[OthelloMove, ...]:
    """Return the placement moves the reference walk finds, in square order."""
    return tuple(OthelloMove(square=sq) for sq in sorted(reference_legal_squares(board)))
