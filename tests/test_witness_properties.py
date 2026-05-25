"""Hypothesis property tests for Othello witness predicates (plan §8 chunk 3).

Per-predicate invariants — sampled across many board states via
:mod:`hypothesis`:

- Corner: corner-landing moves always emit ``CORNER_OCCUPIED``.
- X-square: X-square moves always emit ``X_SQUARE_PLAYED``; never on
  non-X squares.
- Mobility: the magnitude is non-negative (the rule only fires on
  strictly-positive gains, so a fired rule's magnitude is > 0).
- Frontier: the reduction magnitude is strictly positive when the rule
  fires; the exposure magnitude is strictly positive when the rule fires;
  the two rules never fire together on the same (board, move).

Cross-cut:
- Every witness predicate is pure — running it does not mutate the input
  board (verified by comparing the board's bitboards / turn / pass_count
  before and after the predicate call).
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dialectical_othello.board import OthelloBoard, OthelloMove
from dialectical_othello.evidence import (
    ObjectionEvidence,
    ObjectionKind,
    SupportEvidence,
    SupportKind,
)
from dialectical_othello.witnesses import (
    CORNERS,
    X_SQUARES,
    WITNESS_RULES,
)


# --- Board strategies -------------------------------------------------------
#
# A position-strategy that drives the standard opening through a
# pseudo-random sequence of legal moves. Boards generated this way are
# always reachable from the standard starting position and so always
# legitimately represent an Othello game state.


@st.composite
def reachable_boards(draw, max_plies: int = 30) -> OthelloBoard:
    board = OthelloBoard.initial()
    plies = draw(st.integers(min_value=0, max_value=max_plies))
    for _ in range(plies):
        if board.is_terminal():
            return board
        moves = board.legal_moves()
        if not moves:
            return board
        idx = draw(st.integers(min_value=0, max_value=len(moves) - 1))
        board = board.apply(moves[idx])
    return board


# --- Corner / X / C predicates ----------------------------------------------


@given(square=st.sampled_from(sorted(CORNERS)))
def test_corner_predicate_fires_on_every_corner_square(square: int) -> None:
    # The corner predicate is purely syntactic on move.square.
    board = OthelloBoard.initial()
    move = OthelloMove(square=square)
    for rule in WITNESS_RULES:
        if rule.name == "CORNER_OCCUPIED":
            result = rule.predicate(board, move, board)
            assert isinstance(result, SupportEvidence)
            assert result.support_kind is SupportKind.CORNER_OCCUPIED


@given(square=st.sampled_from(sorted(X_SQUARES)))
def test_x_square_predicate_fires_on_every_x_square(square: int) -> None:
    board = OthelloBoard.initial()
    move = OthelloMove(square=square)
    for rule in WITNESS_RULES:
        if rule.name == "X_SQUARE_PLAYED":
            result = rule.predicate(board, move, board)
            assert isinstance(result, ObjectionEvidence)
            assert result.objection_kind is ObjectionKind.X_SQUARE_PLAYED


@given(square=st.integers(min_value=0, max_value=63))
def test_x_square_predicate_never_fires_outside_x_squares(square: int) -> None:
    if square in X_SQUARES:
        return
    board = OthelloBoard.initial()
    move = OthelloMove(square=square)
    for rule in WITNESS_RULES:
        if rule.name == "X_SQUARE_PLAYED":
            result = rule.predicate(board, move, board)
            assert result is None


# --- Magnitude positivity ---------------------------------------------------


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=40)
@given(board=reachable_boards())
def test_mobility_magnitude_is_strictly_positive_when_rule_fires(
    board: OthelloBoard,
) -> None:
    if board.is_terminal():
        return
    moves = board.legal_moves()
    for move in moves:
        if move.is_pass():
            continue
        child = board.apply(move)
        for rule in WITNESS_RULES:
            if rule.name != "MOBILITY_GAIN":
                continue
            result = rule.predicate(board, move, child)
            if result is None:
                continue
            assert isinstance(result, SupportEvidence)
            assert result.support_magnitude is not None
            assert result.support_magnitude > 0


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=40)
@given(board=reachable_boards())
def test_frontier_reduced_and_exposed_are_mutually_exclusive(
    board: OthelloBoard,
) -> None:
    """A single move cannot both strictly-reduce AND strictly-increase the
    mover's frontier count — they are exact opposites on the same delta."""
    if board.is_terminal():
        return
    for move in board.legal_moves():
        if move.is_pass():
            continue
        child = board.apply(move)
        reduced = _run_named("FRONTIER_REDUCED", board, move, child)
        exposed = _run_named("FRONTIER_EXPOSED", board, move, child)
        assert not (reduced is not None and exposed is not None)


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=40)
@given(board=reachable_boards())
def test_frontier_magnitudes_are_strictly_positive_when_fired(
    board: OthelloBoard,
) -> None:
    if board.is_terminal():
        return
    for move in board.legal_moves():
        if move.is_pass():
            continue
        child = board.apply(move)
        reduced = _run_named("FRONTIER_REDUCED", board, move, child)
        if reduced is not None:
            assert isinstance(reduced, SupportEvidence)
            assert reduced.support_magnitude is not None
            assert reduced.support_magnitude > 0
        exposed = _run_named("FRONTIER_EXPOSED", board, move, child)
        if exposed is not None:
            assert isinstance(exposed, ObjectionEvidence)
            assert exposed.objection_magnitude is not None
            assert exposed.objection_magnitude > 0


# --- Cross-cut purity -------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=20)
@given(board=reachable_boards())
def test_every_kind_predicate_pure(board: OthelloBoard) -> None:
    """No witness predicate mutates the input board.

    The dataclass is frozen so mutation would raise; this test additionally
    confirms the snapshot fields remain bit-for-bit identical AND that the
    child board the predicate receives is also unmodified.
    """
    if board.is_terminal():
        return
    moves = board.legal_moves()
    if not moves or moves[0].is_pass():
        return
    for move in moves:
        if move.is_pass():
            continue
        child = board.apply(move)
        snap_parent = (board.black, board.white, board.turn, board.pass_count)
        snap_child = (child.black, child.white, child.turn, child.pass_count)
        for rule in WITNESS_RULES:
            rule.predicate(board, move, child)
        assert (board.black, board.white, board.turn, board.pass_count) == snap_parent
        assert (child.black, child.white, child.turn, child.pass_count) == snap_child


# --- helpers ----------------------------------------------------------------


def _run_named(name: str, board: OthelloBoard, move: OthelloMove, child: OthelloBoard):
    for rule in WITNESS_RULES:
        if rule.name == name:
            return rule.predicate(board, move, child)
    raise KeyError(f"no rule named {name!r}")
