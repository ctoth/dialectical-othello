"""Othello cartridge: search-backend registry.

Registers the Othello alphabeta search backend in the core's
:class:`SearchBackendRegistry`. The shape mirrors
``dialectical_chess.search_backends`` verbatim — Othello v1 only ships
``"alphabeta"`` (chess ships both ``"negamax"`` and ``"alphabeta"`` for
symmetry with checkers; Othello has no use-case for the un-pruned
negamax distinction at v1, so the registry holds the single backend
the plan §7 names).

The ``run`` method does a *real* per-probe search at depth
``settings.search_depth - 1`` on each probe's resulting child board and
returns the probe with the best (lowest, since the child score is
opponent-relative) sub-score. The chess backend stubs the call because
the chess engine drives its own negamax at probe-time inside ``probe.py``;
Othello chunk 4 will follow the same pattern, but the chunk 2
deliverable is to demonstrate the backend genuinely searches. We
therefore implement ``run`` as a working selector rather than a
``return probes[0]`` stub — the play-vs-random smoke needs a real
search to win games.

``settings`` is the cartridge's own engine-settings carrier. Chunk 4
defines ``EngineSettings``; until then ``run`` reads a ``search_depth``
attribute via :func:`getattr` (default 2) so callers can pass any
duck-typed object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dialectical_games.arguments import MoveProbe
from dialectical_games.search_backend import SearchBackend, SearchBackendRegistry

from dialectical_othello.board import OthelloBoard, OthelloMove, _square_from_algebraic
from dialectical_othello.search import INF, SearchTimeout, alphabeta


__all__ = [
    "OthelloAlphaBetaBackend",
    "SEARCH_BACKEND_REGISTRY",
    "select_best_move",
]


def select_best_move(
    board: OthelloBoard,
    depth: int,
    deadline: float | None = None,
) -> OthelloMove:
    """Return the alphabeta-selected best move for ``board`` at ``depth``.

    Convenience wrapper around :func:`alphabeta` for callers (tests,
    play-vs-random harness) that want a move rather than a score. Runs
    a depth-``depth`` search at the root: each legal move is applied,
    the child is searched at ``depth - 1`` with the window negated,
    and the move with the maximum mover-relative score is returned.

    Ties are broken by move order (the first legal move with the top
    score wins). Raises ``ValueError`` on a terminal board (no moves
    to choose). Propagates :class:`SearchTimeout` on deadline overrun.
    """
    if board.is_terminal():
        raise ValueError("select_best_move called on a terminal board")
    moves = board.legal_moves()
    if not moves:
        raise ValueError("select_best_move called with no legal moves")
    best_move = moves[0]
    best_score = -INF
    for move in moves:
        child = board.apply(move)
        score = -alphabeta(child, depth - 1, -INF, INF, deadline)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


@dataclass(frozen=True)
class OthelloAlphaBetaBackend:
    """Othello alphabeta search backend.

    Matches the chess ``NegamaxBackend.run`` signature verbatim
    (``board``, ``probes``, ``settings``, ``deadline``; returns a single
    :class:`MoveProbe`). Reads ``settings.search_depth`` to determine
    the search depth (defaults to 2 if the carrier lacks the attribute
    — chunk 4 wires the real ``EngineSettings``).

    For each probe, applies the probe's move and runs
    :func:`alphabeta` on the child at ``depth - 1`` with a negated
    window. The probe whose child has the *lowest* score (= worst for
    the opponent = best for the mover) is returned. On
    :class:`SearchTimeout` the backend falls back to the best probe
    scored so far, or ``probes[0]`` if none has been scored yet — a
    deadline-respecting backend must always return *some* probe.
    """

    name: str = "alphabeta"

    def run(
        self,
        *,
        board: object,
        probes: tuple[MoveProbe, ...],
        settings: Any,
        deadline: float | None,
    ) -> MoveProbe:
        if not probes:
            raise ValueError("OthelloAlphaBetaBackend.run requires at least one probe")
        if not isinstance(board, OthelloBoard):
            raise TypeError(
                "OthelloAlphaBetaBackend.run requires an OthelloBoard, "
                f"got {type(board).__name__}"
            )
        depth: int = int(getattr(settings, "search_depth", 2))
        best_probe = probes[0]
        best_score = -INF
        for probe in probes:
            move = _move_for_probe(probe)
            try:
                child = board.apply(move)
                score = -alphabeta(child, depth - 1, -INF, INF, deadline)
            except SearchTimeout:
                # Deadline blew during this probe's search. Return the
                # best probe scored so far (or ``probes[0]`` if this is
                # the first probe and nothing has been scored yet).
                return best_probe
            if score > best_score:
                best_score = score
                best_probe = probe
        return best_probe


def _move_for_probe(probe: MoveProbe) -> OthelloMove:
    """Reconstruct the :class:`OthelloMove` from ``probe.move_id``.

    The core ``MoveProbe`` carries only the string ``move_id`` (the
    cartridge's stable move identifier). For Othello that string is
    either ``"pass"`` or two-character algebraic notation (``"d3"``,
    ``"e6"``, ...). Chunk 4 will introduce a typed Othello ``MoveProbe``
    subclass that carries the move directly; until then the backend
    reconstructs from the id.
    """
    move_id = probe.move_id
    if move_id == "pass":
        return OthelloMove.pass_move()
    return OthelloMove(square=_square_from_algebraic(move_id))


SEARCH_BACKEND_REGISTRY: SearchBackendRegistry = SearchBackendRegistry()
_backend: SearchBackend = OthelloAlphaBetaBackend()
SEARCH_BACKEND_REGISTRY.register(_backend)
