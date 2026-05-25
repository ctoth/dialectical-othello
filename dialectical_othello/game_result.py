"""``OthelloGameResult`` — the GameResult-Protocol implementation for Othello.

Satisfies :class:`dialectical_games.game_result.GameResult` (game_result.py:18-42):
``outcome`` is a cartridge-owned string (the core compares only by equality),
``moves`` is the move sequence, ``positions`` is every board from start to
terminal (``len(moves) + 1`` entries).

Outcome strings per plan §2:

- ``"B+N"`` — Black wins by N discs (1 <= N <= 64).
- ``"W+N"`` — White wins by N discs.
- ``"draw"`` — 32-32.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialectical_othello.board import OthelloBoard, OthelloMove


def format_outcome(black_discs: int, white_discs: int) -> str:
    """Render the Othello outcome string from the final disc counts.

    Either ``"B+N"`` / ``"W+N"`` / ``"draw"``. ``N`` is the disc-count
    margin of victory (always >= 1 when there is a winner).
    """
    if black_discs > white_discs:
        return f"B+{black_discs - white_discs}"
    if white_discs > black_discs:
        return f"W+{white_discs - black_discs}"
    return "draw"


@dataclass(frozen=True)
class OthelloGameResult:
    """The result of one Othello game (plan §2)."""

    outcome: str
    moves: tuple[OthelloMove, ...]
    positions: tuple[OthelloBoard, ...]

    @classmethod
    def from_terminal(
        cls,
        moves: tuple[OthelloMove, ...],
        positions: tuple[OthelloBoard, ...],
    ) -> OthelloGameResult:
        """Build a result from a played-out game.

        ``positions[-1]`` must be terminal (``is_terminal() == True``).
        ``len(positions) == len(moves) + 1`` is required by the GameResult
        Protocol; both are validated.
        """
        if len(positions) != len(moves) + 1:
            raise ValueError(
                f"positions length {len(positions)} must equal moves "
                f"length {len(moves)} + 1"
            )
        if not positions:
            raise ValueError("positions must be non-empty")
        final = positions[-1]
        if not final.is_terminal():
            raise ValueError("final position must be terminal (pass_count >= 2)")
        black, white = final.disc_counts()
        return cls(outcome=format_outcome(black, white), moves=moves, positions=positions)
