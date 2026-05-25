"""Property tests for ``OthelloBoard`` (plan chunk 1).

Five invariants from the chunk-1 spec:

1. The starting position has exactly 4 legal moves for Black.
2. Serialisation round-trips for any reachable state.
3. Applying any legal move flips at least one disc.
4. Terminal game outcomes always sum to 64 (B + W + empty).
5. The bitboard ``legal_moves()`` agrees square-for-square with the slow
   direction-walk reference implementation.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from dialectical_othello.board import OthelloBoard, OthelloMove
from dialectical_othello.game_result import OthelloGameResult, format_outcome
from tests._legal_moves_reference import reference_legal_squares


# --- Helpers ----------------------------------------------------------------


def _play_random_game(seed: int, max_plies: int = 80) -> tuple[
    OthelloBoard, list[OthelloMove], list[OthelloBoard]
]:
    """Play a deterministic random game; return (final, moves, positions).

    Stops at terminal or ``max_plies``. Auto-pass is handled inside
    :meth:`OthelloBoard.apply`, so the only moves chosen here are disc
    placements (or an explicit pass move if the from-fen state demands it,
    which never happens for the initial-position start used here).
    """
    rng = random.Random(seed)
    board = OthelloBoard.initial()
    moves: list[OthelloMove] = []
    positions: list[OthelloBoard] = [board]
    for _ in range(max_plies):
        if board.is_terminal():
            break
        legal = board.legal_moves()
        if not legal:
            break
        move = rng.choice(legal)
        board = board.apply(move)
        moves.append(move)
        positions.append(board)
    return board, moves, positions


def _reachable_state(seed: int, plies: int) -> OthelloBoard:
    """Play up to ``plies`` random moves; return the resulting (possibly
    not-yet-terminal) state."""
    rng = random.Random(seed)
    board = OthelloBoard.initial()
    for _ in range(plies):
        if board.is_terminal():
            return board
        legal = board.legal_moves()
        if not legal:
            return board
        board = board.apply(rng.choice(legal))
    return board


# --- 1. Starting position has 4 legal moves for Black -----------------------


def test_starting_position_has_four_legal_moves() -> None:
    board = OthelloBoard.initial()
    assert board.turn == "B"
    moves = board.legal_moves()
    assert len(moves) == 4
    # The canonical four — d3, c4, f5, e6 — are the only squares that
    # bracket a white disc on the opening position.
    assert {m.move_id() for m in moves} == {"d3", "c4", "f5", "e6"}


# --- 2. Serialisation round-trip --------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_serialization_round_trip(seed: int, plies: int) -> None:
    board = _reachable_state(seed, plies)
    fen = board.to_fen()
    restored = OthelloBoard.from_fen(fen)
    assert restored == board
    # ``to_fen`` of the restored board must produce the same string.
    assert restored.to_fen() == fen


# --- 3. Applying a legal placement flips at least one disc ------------------


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_apply_flips_at_least_one_disc(seed: int, plies: int) -> None:
    board = _reachable_state(seed, plies)
    if board.is_terminal():
        return
    legal = board.legal_moves()
    if not legal:
        return
    # Pick the first placement move (if any). The reachable-state generator
    # never lands on a forced-pass-only board because :meth:`apply`
    # auto-passes, but a from-fen state could; for this property we only
    # care about placement moves.
    placements = [m for m in legal if not m.is_pass()]
    if not placements:
        return
    for move in placements:
        before = board
        after = board.apply(move)
        # Compute the symmetric difference of opponent bitboards: the
        # squares that flipped from opponent to mover.
        if before.turn == "B":
            flipped = before.white & ~after.white
        else:
            flipped = before.black & ~after.black
        assert flipped.bit_count() >= 1, (
            f"move {move.move_id()} on {before.to_fen()!r} flipped zero discs"
        )


# --- 4. Terminal game outcome sums to 64 ------------------------------------


@pytest.mark.parametrize("seed", list(range(10)))
def test_terminal_game_outcome_sums_to_64(seed: int) -> None:
    final, moves, positions = _play_random_game(seed, max_plies=80)
    # A random game on the initial position always terminates within 64
    # plies (every placement fills a square; the board is 64 squares with 4
    # pre-placed, so at most 60 placements + at most a few passes).
    assert final.is_terminal(), (
        f"seed {seed}: game did not terminate after 80 plies"
    )
    black, white = final.disc_counts()
    empty = final.empty_count
    assert black + white + empty == 64
    result = OthelloGameResult.from_terminal(tuple(moves), tuple(positions))
    expected_outcome = format_outcome(black, white)
    assert result.outcome == expected_outcome
    # GameResult Protocol: len(positions) == len(moves) + 1.
    assert len(result.positions) == len(result.moves) + 1


# --- 5. Bitboard legal_moves matches slow reference -------------------------


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    plies=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_legal_move_count_matches_direction_walk(seed: int, plies: int) -> None:
    board = _reachable_state(seed, plies)
    expected = reference_legal_squares(board)
    actual = board.legal_moves()
    if expected:
        # The board has at least one placement → bitboard returns those
        # placements, no pass.
        actual_squares = {m.square for m in actual if not m.is_pass()}
        assert actual_squares == expected
        assert not any(m.is_pass() for m in actual)
    else:
        # The bitboard returns either the empty tuple (terminal) or a
        # singleton pass (forced pass on a non-terminal board).
        assert all(m.is_pass() for m in actual)


# --- Bonus: explicit starting-FEN sanity ------------------------------------


def test_initial_to_fen_starting_shape() -> None:
    """Concrete starting-FEN literal — anchors the format documentation."""
    board = OthelloBoard.initial()
    expected = "......../......../......../...BW.../...WB.../......../......../........ B 0"
    assert board.to_fen() == expected
