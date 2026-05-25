"""Genuine starting-position smoke test for ``OthelloBoard`` (chunk 1).

Replaces the original import-only smoke. Verifies:

- The starting position has 4 legal moves for Black.
- Applying one of those moves yields a White-to-move board with the
  standard 3-reply count (Black plays one of d3/c4/f5/e6 → White has
  exactly 3 legal replies).
- ``to_fen()`` returns the documented 8-row + turn + pass-count string.
"""

from __future__ import annotations

from dialectical_games.board import Board, Move

from dialectical_othello.board import OthelloBoard, OthelloMove


def test_dialectical_games_protocols_importable() -> None:
    """Sanity: the upstream Protocols still resolve from the pinned core."""
    assert Board is not None
    assert Move is not None


def test_starting_position_smoke() -> None:
    board = OthelloBoard.initial()
    assert board.turn == "B"
    legal = board.legal_moves()
    assert len(legal) == 4
    move_ids = sorted(m.move_id() for m in legal)
    assert move_ids == ["c4", "d3", "e6", "f5"]
    # Pick d3 (a canonical opening): White then has exactly 3 legal replies
    # (the standard Othello opening tree — every first move leads to a
    # 3-reply position by symmetry).
    d3 = next(m for m in legal if m.move_id() == "d3")
    after_d3 = board.apply(d3)
    assert after_d3.turn == "W"
    white_replies = after_d3.legal_moves()
    assert len(white_replies) == 3
    # to_fen returns the documented shape.
    fen = board.to_fen()
    assert fen == "......../......../......../...BW.../...WB.../......../......../........ B 0"
    # And applying d3 produces a serialisable resulting state.
    after_fen = after_d3.to_fen()
    parts = after_fen.split()
    assert len(parts) == 3
    rows = parts[0].split("/")
    assert len(rows) == 8 and all(len(r) == 8 for r in rows)
    assert parts[1] == "W"
    assert parts[2] == "0"


def test_pass_move_id() -> None:
    """``OthelloMove.move_id()`` returns ``"pass"`` for a pass move."""
    assert OthelloMove.pass_move().move_id() == "pass"
