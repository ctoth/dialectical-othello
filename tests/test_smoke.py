"""Smoke test: proves the ``dialectical-games`` git pin resolves and imports.

This test does not exercise any Othello logic; it only verifies that the
``Board`` and ``Move`` Protocols from the shared core are importable in this
environment. When the Othello cartridge is implemented, real contract tests
will replace / augment this.
"""

from dialectical_games.board import Board, Move


def test_dialectical_games_protocols_importable() -> None:
    assert Board is not None
    assert Move is not None
