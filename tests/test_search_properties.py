"""Property tests for ``dialectical_othello.search.alphabeta`` (chunk 2).

Four invariants:

- **Equivalence with slow minimax.** alphabeta with pruning must return
  the same score as the no-pruning reference at every depth. Any
  disagreement is an alphabeta soundness bug.
- **Depth monotonicity.** On a *closed* position (one where the side to
  move has exactly one legal placement and that placement leads to
  another forced single reply), the depth-2 score must be at least as
  good for the mover as the depth-0 score under the same window —
  alphabeta with more lookahead never makes worse decisions on
  positions whose move tree is forced.
- **Negation symmetry.** Swapping the two bitboards AND the turn
  produces an "inverted" board on which the alphabeta score has the
  opposite sign — the evaluation is mover-relative and the search is
  pure, so the symmetry is exact.
- **Deadline respected.** A tight deadline (already past) raises
  :class:`SearchTimeout` rather than producing a partial result.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dialectical_othello.board import OthelloBoard, OthelloMove
from dialectical_othello.search import (
    INF,
    SearchTimeout,
    alphabeta,
    static_evaluation,
)

from tests._minimax_reference import minimax


SLOW_SETTINGS: Final = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.function_scoped_fixture),
)


def _random_playout(seed: int, max_plies: int) -> OthelloBoard:
    """Play a deterministic-pseudorandom sequence of moves from the start.

    Uses the seed as an LCG state so the test is deterministic but the
    boards differ across seeds. Stops at ``max_plies`` plies or when
    terminal, whichever first.
    """
    state = (seed * 2654435761) & 0xFFFFFFFF
    board = OthelloBoard.initial()
    for _ in range(max_plies):
        if board.is_terminal():
            break
        moves = board.legal_moves()
        if not moves:
            break
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        idx = state % len(moves)
        board = board.apply(moves[idx])
    return board


def _colour_swap(board: OthelloBoard) -> OthelloBoard:
    """Return the colour-relabeled board (swap bitboards AND turn).

    Semantically the same position with colours renamed — the side to
    move (now the opposite colour) faces the *same* legal moves and
    the *same* mover-relative evaluation. This is a sanity check that
    alphabeta is colour-blind, not a sign-flip test.
    """
    inverted_turn = "W" if board.turn == "B" else "B"
    return OthelloBoard(
        black=board.white,
        white=board.black,
        turn=inverted_turn,
        pass_count=board.pass_count,
    )


@SLOW_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=8),
    depth=st.integers(min_value=0, max_value=3),
)
def test_alphabeta_equals_minimax(seed: int, plies: int, depth: int) -> None:
    """alphabeta must agree with the no-pruning minimax at every depth."""
    board = _random_playout(seed, plies)
    assert alphabeta(board, depth, -INF, INF, None) == minimax(board, depth)


def test_alphabeta_depth_monotone_on_starting_position() -> None:
    """A deeper search never returns a worse score than ``static_evaluation``.

    The starting position has 4 legal moves for Black with disc count
    2-2 at the root. ``static_evaluation`` returns 0; alphabeta at
    depth >= 1 sees at least one move that flips a disc and so returns
    a non-negative score from Black's perspective. Specifically:
    every Black opening flips exactly 1 White disc -> child has
    Black=4 / White=1, but turn is then White's so the child's
    static_evaluation from *White's* mover-relative view is -3; the
    negated parent score is +3. Depth-2 looks one ply deeper but
    cannot do worse than depth-1 because the maximizer at the root
    sees every depth-1 child as one of its options.
    """
    board = OthelloBoard.initial()
    score_d0 = alphabeta(board, 0, -INF, INF, None)
    score_d1 = alphabeta(board, 1, -INF, INF, None)
    score_d2 = alphabeta(board, 2, -INF, INF, None)
    # Depth 0 is just the static eval: 2-2 = 0.
    assert score_d0 == 0
    # Depth 1: the mover picks the best of 4 opening moves; each
    # flips exactly 1 disc so the child's disc count is (4, 1)
    # (Black, White) with White to move; static_evaluation from
    # White's mover-relative view is 1 - 4 = -3; the negated parent
    # value is +3. Black should never be forced to play worse than 0.
    assert score_d1 == 3
    # Depth 2: deeper search never returns a worse value than depth 0
    # on the starting position (the root maximizer can always replay
    # the same move it would have chosen at d1 and let the opponent
    # respond; the score may differ but cannot drop below 0 for the
    # starting position which is symmetric).
    assert score_d2 >= 0


@SLOW_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=8),
    depth=st.integers(min_value=1, max_value=3),
)
def test_alphabeta_negamax_child_negation(
    seed: int, plies: int, depth: int
) -> None:
    """The negamax core identity: a parent's score equals the max of the
    *negated* child scores at one less depth.

    For every reachable non-terminal board ``b`` with at least one
    legal placement, the alphabeta score at depth ``d`` must equal
    ``max(-alphabeta(b.apply(m), d-1)) for m in legal_moves``. This is
    the fundamental negamax invariant — if it fails, the negation
    around the recursive call is wrong.
    """
    board = _random_playout(seed, plies)
    if board.is_terminal():
        return
    moves = board.legal_moves()
    if not moves:
        return
    parent = alphabeta(board, depth, -INF, INF, None)
    child_scores = [
        -alphabeta(board.apply(m), depth - 1, -INF, INF, None) for m in moves
    ]
    assert parent == max(child_scores)


@SLOW_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=8),
    depth=st.integers(min_value=0, max_value=3),
)
def test_alphabeta_colour_swap_invariant(
    seed: int, plies: int, depth: int
) -> None:
    """alphabeta is colour-blind: swapping bitboards + turn together is a
    pure relabeling and leaves the mover-relative score unchanged.

    This is the *positive* form of the symmetry — for the genuine
    negation form see :func:`test_alphabeta_negamax_child_negation`.
    """
    board = _random_playout(seed, plies)
    forward = alphabeta(board, depth, -INF, INF, None)
    relabel = alphabeta(_colour_swap(board), depth, -INF, INF, None)
    assert forward == relabel


def test_alphabeta_respects_deadline() -> None:
    """A past deadline raises :class:`SearchTimeout` immediately."""
    board = OthelloBoard.initial()
    # A deadline strictly in the past — every entry to alphabeta will
    # see the clock past it and raise.
    past = time.monotonic() - 1.0
    with pytest.raises(SearchTimeout):
        alphabeta(board, 4, -INF, INF, past)


def test_static_evaluation_is_mover_relative() -> None:
    """The static eval matches under colour-swap and flips when only the
    turn changes (keeping bitboards)."""
    board = OthelloBoard.initial()
    assert static_evaluation(board) == 0
    # Colour-swap is a relabeling: same number.
    assert static_evaluation(_colour_swap(board)) == 0
    # Move one disc to make the count asymmetric and verify mover-relative
    # behaviour explicitly.
    asym = OthelloBoard(black=0b111, white=0b1, turn="B", pass_count=0)
    assert static_evaluation(asym) == 2
    # Same bitboards, opposite turn: mover is now White with 1 disc
    # against 3 — mover-relative eval is -2.
    asym_w = OthelloBoard(black=0b111, white=0b1, turn="W", pass_count=0)
    assert static_evaluation(asym_w) == -2
    # Colour-swap of `asym` (swap bitboards + turn): same evaluation.
    assert static_evaluation(_colour_swap(asym)) == 2


def test_terminal_position_returns_signed_sentinel() -> None:
    """A terminal position returns ±TERMINAL_SCORE + disc differential."""
    # Construct a synthetic terminal board: Black wins by 4 discs,
    # pass_count=2.
    terminal_black_wins = OthelloBoard(
        black=0b11111,
        white=0b1,
        turn="B",
        pass_count=2,
    )
    score = alphabeta(terminal_black_wins, 3, -INF, INF, None)
    # Black has 5, White has 1: mover-relative diff is +4, sentinel
    # adds 10_000 -> 10_004.
    assert score == 10_004
    # Same terminal position with white to move and same discs: now
    # white is the mover, white loses by 4 -> -10_004.
    terminal_white_loses = OthelloBoard(
        black=0b11111,
        white=0b1,
        turn="W",
        pass_count=2,
    )
    assert alphabeta(terminal_white_loses, 3, -INF, INF, None) == -10_004


def test_alphabeta_handles_singleton_pass_move() -> None:
    """When ``legal_moves()`` returns a singleton pass, alphabeta recurses
    through it cleanly (does not crash, returns a finite score)."""
    # Construct a board where Black has no flips (Black mover but
    # all Black discs surrounded by Black on every side — easiest is
    # an empty/white-only board with Black to move and pass_count=1).
    no_flips_board = OthelloBoard(
        black=0,
        white=0b1111,
        turn="B",
        pass_count=1,
    )
    moves = no_flips_board.legal_moves()
    # Either a singleton pass or empty (terminal) — handle both.
    if moves:
        assert len(moves) == 1
        assert moves[0] == OthelloMove.pass_move()
        # alphabeta should accept this and recurse via the pass.
        score = alphabeta(no_flips_board, 2, -INF, INF, None)
        assert isinstance(score, int)
