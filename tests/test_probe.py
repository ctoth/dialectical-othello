"""Tests for the :mod:`dialectical_othello.probe` driver.

The probe driver walks ``WITNESS_RULES`` per legal move, builds typed
evidence + core labels, and returns one :class:`OthelloMoveProbe` per move.
These tests pin: one-probe-per-legal-move, core-label translation,
contested-bit computation, child_eval orientation.
"""

from __future__ import annotations

from dialectical_othello.board import OthelloBoard
from dialectical_othello.probe import (
    OthelloMoveProbe,
    probe_moves,
)


def test_probe_moves_one_probe_per_legal_move() -> None:
    board = OthelloBoard.initial()
    probes = probe_moves(board)
    moves = board.legal_moves()
    assert len(probes) == len(moves)
    assert tuple(p.move_id for p in probes) == tuple(m.move_id() for m in moves)


def test_probe_moves_terminal_returns_empty() -> None:
    # Fully passed terminal board: no legal moves -> no probes.
    board = OthelloBoard.initial()
    # Drive the board to a state with no legal flips by simulating two
    # passes via the substrate ``apply`` is not the cleanest path; instead
    # construct a fully-occupied board where neither side can move.
    # Construct: all 64 squares Black -> White has no legal flip, Black
    # has no legal flip either (no empty square). pass_count=2 == terminal.
    full = OthelloBoard(black=(1 << 64) - 1, white=0, turn="B", pass_count=2)
    probes = probe_moves(full)
    assert probes == ()


def test_probe_carries_core_labels_for_typed_evidence() -> None:
    # On the opening, every Black move fires obj:frontier:* and
    # pro:cramps_opponent:*, so each probe's core labels include both.
    board = OthelloBoard.initial()
    for probe in probe_moves(board):
        assert any(label.startswith("obj:frontier:") for label in probe.objections), probe
        assert any(label.startswith("pro:cramps_opponent:") for label in probe.reasons), probe


def test_probe_is_othello_move_probe_subclass() -> None:
    board = OthelloBoard.initial()
    for probe in probe_moves(board):
        assert isinstance(probe, OthelloMoveProbe)


def test_probe_typed_evidence_matches_core_labels_count() -> None:
    """For each probe, the number of typed evidence items is >= the number
    of core labels (the translator drops typed items it can't map; chunk 3
    has a translator entry for every implemented rule so the counts are
    equal in practice)."""
    board = OthelloBoard.initial()
    for probe in probe_moves(board):
        assert len(probe.reason_evidence) >= len(probe.reasons)
        assert len(probe.objection_evidence) >= len(probe.objections)


def test_probe_contested_flag_set_when_both_pro_and_obj_present() -> None:
    # On the opening, every Black move has BOTH a pro (cramps_opponent)
    # AND an obj (frontier) — so every probe is contested.
    board = OthelloBoard.initial()
    for probe in probe_moves(board):
        assert probe.contested is True


def test_probe_child_eval_is_parent_mover_relative() -> None:
    # On the opening, every Black move flips exactly one White disc -> the
    # post-move count is Black:4, White:1. The child's mover is White, so
    # static_evaluation(child) = white-black = 1-4 = -3. The probe's
    # child_eval negates that to the parent-mover (Black) perspective: +3.
    board = OthelloBoard.initial()
    for probe in probe_moves(board):
        assert probe.child_eval == 3
