"""End-to-end smoke for :class:`OthelloEngine` (chunk 4, plan §8 D4).

Drives the full depth-0 pipeline (probes -> graded graph -> decider)
through :meth:`OthelloEngine.analyze` on the starting Othello position
and confirms the chosen move is one of the four legal opening moves.
Then plays a 2-ply self-play game (four half-moves) to confirm the
pipeline composes across positions.
"""

from __future__ import annotations

from dialectical_othello.board import OthelloBoard
from dialectical_othello.engine import (
    OthelloEngine,
    OthelloEngineAnalysis,
    OthelloEngineSettings,
)
from dialectical_othello.probe import OthelloMoveProbe


# The four canonical Othello opening moves available to Black from the
# standard starting position (any disc placement that brackets the
# centre White disc): d3, c4, f5, e6.
_LEGAL_OPENINGS: frozenset[str] = frozenset({"d3", "c4", "f5", "e6"})


def test_analyze_starting_position_returns_legal_opening() -> None:
    """The engine returns an analysis with one probe per legal opening move
    and a decision pointing at one of the four canonical openings."""
    board = OthelloBoard.initial()
    engine = OthelloEngine(settings=OthelloEngineSettings(search_depth=2))
    analysis = engine.analyze(board)

    assert isinstance(analysis, OthelloEngineAnalysis)
    probe_ids = {probe.move_id for probe in analysis.probes}
    assert probe_ids == _LEGAL_OPENINGS
    # Every probe is the cartridge-typed subclass with the typed move.
    for probe in analysis.probes:
        assert isinstance(probe, OthelloMoveProbe)
        assert probe.move.move_id() == probe.move_id

    decision = analysis.decision
    assert decision.selected is not None
    assert decision.move_id in _LEGAL_OPENINGS
    assert decision.selected.move_id == decision.move_id


def test_two_ply_self_play_completes_without_error() -> None:
    """Alternate sides for four half-moves; every analyze yields a legal
    move and the resulting board remains valid (non-terminal at end)."""
    board = OthelloBoard.initial()
    engine = OthelloEngine(settings=OthelloEngineSettings(search_depth=2))

    half_moves_played: list[str] = []
    for _ply in range(4):
        analysis = engine.analyze(board)
        decision = analysis.decision
        assert decision.selected is not None, (
            f"engine returned no decision at ply {_ply}; "
            f"board={board.to_fen()!r}"
        )
        chosen = decision.selected
        assert isinstance(chosen, OthelloMoveProbe)
        # The chosen probe's typed move must be among the legal moves.
        legal_ids = {m.move_id() for m in board.legal_moves()}
        assert chosen.move_id in legal_ids, (
            f"chosen {chosen.move_id!r} not in legal set {legal_ids!r}"
        )
        board = board.apply(chosen.move)
        half_moves_played.append(chosen.move_id)

    assert len(half_moves_played) == 4
    # Four half-moves on a starting Othello position cannot reach
    # terminal (the earliest known full-board terminal is many plies
    # away); the board remains non-terminal so the game can continue.
    assert not board.is_terminal()


def test_analyze_terminal_position_returns_null_decision() -> None:
    """A board whose ``pass_count`` is 2 has no legal moves; the engine
    returns an analysis whose decision is null and probe list empty."""
    # Construct a contrived terminal-by-double-pass position: an empty
    # board with pass_count=2 is the simplest construction.
    terminal = OthelloBoard(black=0, white=0, turn="B", pass_count=2)
    engine = OthelloEngine()
    analysis = engine.analyze(terminal)
    assert analysis.probes == ()
    assert analysis.decision.selected is None
    assert analysis.decision.move_id == ""
