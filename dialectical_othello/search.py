"""Alphabeta search primitive over the Othello disc-differential evaluation.

Chunk 2 (plan §7, §8). The evaluation is the **mover-relative** disc-count
differential — there is no other heuristic here at v1; the entire job of this
module is to bring the alphabeta machinery online so the Cartridge can run
real searches at probe time later (chunk 4).

Design notes:

- **Negamax form.** A single recursive function with `(alpha, beta)`
  window flipped + negated at each recursion. The chess reference
  (``dialectical_chess.search.negamax``) is more elaborate (caches,
  position-history, reply analysis) — Othello does not need any of that:
  no draw-by-repetition, no transposition table at v1.
- **Auto-pass via ``apply()``.** ``OthelloBoard.apply()`` already
  advances past forced passes (board.py:282-360). So when
  ``board.legal_moves()`` returns the singleton pass move
  ``OthelloMove(square=None)`` — which can only happen on a freshly-
  constructed board, never on a board produced by ``apply()`` — we
  *apply that pass* and recurse one ply deeper. We never need a
  separate ``passed()`` helper because the substrate already does the
  bookkeeping.
- **Terminal sentinel ±10_000.** Bigger than the 64-disc max
  differential by a comfortable factor so any alphabeta window
  comparison can distinguish "wins by N discs" from "merely good by N
  discs."
- **Deadline.** ``time.monotonic()`` check at every recursive entry.
  Exceeding the deadline raises :class:`SearchTimeout`. ``None``
  deadline means unlimited.
"""

from __future__ import annotations

import time

from dialectical_othello.board import OthelloBoard


__all__ = [
    "INF",
    "TERMINAL_SCORE",
    "SearchTimeout",
    "alphabeta",
    "static_evaluation",
]


#: Larger than any disc differential (max 64) so it dominates window
#: comparisons but does not approach Python int overflow concerns (Python
#: ints are arbitrary precision; the bound is purely a clarity choice).
TERMINAL_SCORE = 10_000

#: Alphabeta window bound. Bigger than ``TERMINAL_SCORE`` so the initial
#: window strictly contains every reachable score.
INF = 1_000_000


class SearchTimeout(Exception):
    """Raised when ``alphabeta`` runs past its monotonic-clock deadline.

    A backend that wants to cap wall-clock time passes a
    ``deadline = time.monotonic() + budget`` to :func:`alphabeta`; the
    function checks the clock at every recursive entry and raises this
    exception the moment the deadline has passed. Callers that ignore
    the deadline simply pass ``None``.
    """


def static_evaluation(board: OthelloBoard) -> int:
    """Return the mover-relative disc-count differential.

    Positive means the side to move is ahead. Symmetric: swapping the
    bitboards and the turn flips the sign exactly. For a terminal
    position the magnitude is the absolute disc-count gap; for an
    in-progress position it is the running tally. The alphabeta caller
    overrides this with :data:`TERMINAL_SCORE` (signed) on terminal
    positions so the result clearly separates "wins" from "good."
    """
    black, white = board.disc_counts()
    if board.turn == "B":
        return black - white
    return white - black


def _terminal_value(board: OthelloBoard) -> int:
    """Mover-relative ±TERMINAL_SCORE for a terminal position, 0 on a tie.

    The magnitude carries the disc differential so deeper-but-equal-sign
    terminal positions distinguish: e.g. a "win by 30" beats a "win by
    2" if both are forced from the root. The sentinel ``TERMINAL_SCORE``
    is added on top of the differential so a *terminal* win always
    dominates any in-progress score in the same window.
    """
    diff = static_evaluation(board)
    if diff > 0:
        return TERMINAL_SCORE + diff
    if diff < 0:
        return -TERMINAL_SCORE + diff
    return 0


def alphabeta(
    board: OthelloBoard,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float | None,
) -> int:
    """Return the mover-relative alphabeta score of ``board`` at ``depth``.

    Negamax form. ``alpha`` and ``beta`` are the current window from the
    mover's perspective; the recursion flips and negates them. ``depth``
    is plies remaining; ``depth == 0`` returns the static evaluation.
    Terminal positions (``pass_count >= 2``) return the signed
    :data:`TERMINAL_SCORE` plus the disc differential regardless of
    depth.

    Raises :class:`SearchTimeout` if ``deadline`` is not ``None`` and the
    monotonic clock has passed it on entry. The check happens *before*
    any work so a caller that sets a deadline already in the past gets a
    timeout immediately rather than a partial result.
    """
    if deadline is not None and time.monotonic() >= deadline:
        raise SearchTimeout()
    if board.is_terminal():
        return _terminal_value(board)
    if depth <= 0:
        return static_evaluation(board)
    moves = board.legal_moves()
    if not moves:
        # Shouldn't happen on a non-terminal board (is_terminal would be
        # true), but defensive: treat as terminal-by-no-moves.
        return _terminal_value(board)
    # ``legal_moves()`` returns a singleton pass on freshly-constructed
    # boards where the mover has no disc-flip; ``apply()``-produced
    # boards have already absorbed forced passes. Either way: iterate
    # ``moves`` uniformly — ``apply()`` accepts the pass move when it is
    # legal and the next-board logic stays consistent.
    best = -INF
    for move in moves:
        child = board.apply(move)
        score = -alphabeta(child, depth - 1, -beta, -alpha, deadline)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best
