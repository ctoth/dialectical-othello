"""Play-vs-random smoke for the chunk 2 alphabeta backend.

The dialectical reasoning is not online until chunk 4 — what chunk 2
must demonstrate is that the *search itself* is strong enough that a
depth-2 alphabeta convincingly beats uniform-random move selection.
If alphabeta-d2 cannot win the majority of games against pure random,
the eval or the search is broken.

A single game from a fixed seed is too noisy to be diagnostic (random
sometimes stumbles into a corner-win), so we play a small seeded
match: 20 games, alphabeta as Black, random as White, fixed
per-game seeds. Required: alphabeta wins >= 13 of 20.

Empirically (manual probe) alphabeta-d2 wins ~17-19 of 20 against
uniform random — the threshold of 13/20 is comfortable headroom; if
the smoke ever drops below it, something has regressed.
"""

from __future__ import annotations

import random

import pytest

from dialectical_othello.board import OthelloBoard, OthelloMove
from dialectical_othello.search_backends import select_best_move


def _random_move(board: OthelloBoard, rng: random.Random) -> OthelloMove:
    """Pick a uniformly-random legal move (including pass if forced)."""
    moves = board.legal_moves()
    return rng.choice(list(moves))


def _play_game(seed: int, alphabeta_depth: int, alphabeta_is_black: bool) -> str:
    """Play one game; return ``"B"``, ``"W"``, or ``"draw"`` for the winner.

    Alphabeta plays one colour (Black if ``alphabeta_is_black``, else
    White); the other colour plays uniformly random with ``rng = Random(seed)``.
    """
    rng = random.Random(seed)
    board = OthelloBoard.initial()
    alphabeta_colour = "B" if alphabeta_is_black else "W"
    # Cap plies to a generous ceiling — Othello tops out at 60 placements,
    # and pass moves resolve into terminal via ``apply``. The cap is pure
    # paranoia against an infinite-loop bug.
    for _ in range(200):
        if board.is_terminal():
            break
        if board.turn == alphabeta_colour:
            move = select_best_move(board, alphabeta_depth, deadline=None)
        else:
            move = _random_move(board, rng)
        board = board.apply(move)
    black, white = board.disc_counts()
    if black > white:
        return "B"
    if white > black:
        return "W"
    return "draw"


@pytest.mark.differential
def test_alphabeta_d2_beats_random_in_majority() -> None:
    """alphabeta-d2 as Black wins >= 13 of 20 vs uniform-random White."""
    wins = 0
    losses = 0
    draws = 0
    for seed in range(20):
        winner = _play_game(seed=seed, alphabeta_depth=2, alphabeta_is_black=True)
        if winner == "B":
            wins += 1
        elif winner == "W":
            losses += 1
        else:
            draws += 1
    # 13/20 is comfortable headroom over the 10/20 50% mark; alphabeta-d2
    # ought to win most of these. If it drops below 13 the search or eval
    # has regressed.
    assert wins >= 13, (
        f"alphabeta-d2 vs random: {wins} wins / {losses} losses / "
        f"{draws} draws (need >= 13 wins)"
    )
