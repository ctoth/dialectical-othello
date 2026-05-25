"""Othello cartridge: ``Cartridge`` Protocol impl + engine driver.

The Othello cartridge implements the core
:class:`dialectical_games.engine.Cartridge` Protocol (``probe_moves`` +
``make_graded_policy``) and drives moves through the core orchestrator
:func:`dialectical_games.engine.analyze`.

Plan §8 (chunk 4) + scout §5 pin: **no post-decision hook at v1**. The
endgame-solver hook is v2 work, gated on the empty-square count and on a
deadline-respecting full-solve. Until then ``analyze`` calls the core
orchestrator with ``post_decision=None`` and the orchestrator's depth-0
pipeline is the entire engine output: probes -> graded graph -> decider.

The engine carries an :class:`OthelloEngineSettings` dataclass; the only
fields the core orchestrator reads are ``search_backend`` (the registry
key) and ``deadline``. The full settings carrier is threaded onto the
core ``EngineSettings.cartridge_settings`` slot so a future post-decision
hook (chunk H+) can read its own knobs without the orchestrator caring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from dialectical_games.arguments import MoveProbe as CoreMoveProbe
from dialectical_games.engine import (
    EngineAnalysis as CoreEngineAnalysis,
    EngineSettings as CoreEngineSettings,
    analyze as core_analyze,
)

from dialectical_othello.board import OthelloBoard
from dialectical_othello.graded_policy import OthelloGradedPolicy
from dialectical_othello.probe import OthelloMoveProbe, probe_moves


__all__ = [
    "OthelloEngine",
    "OthelloEngineAnalysis",
    "OthelloEngineDecision",
    "OthelloEngineSettings",
]


@dataclass(frozen=True)
class OthelloEngineSettings:
    """Othello cartridge engine settings.

    Carried opaquely on the core ``EngineSettings.cartridge_settings``
    slot. The two fields the core orchestrator reads -- ``search_backend``
    and ``deadline`` -- are also forwarded onto the core settings.

    ``search_backend`` defaults to ``"alphabeta"``: the single backend
    chunk 2 registered in :data:`dialectical_othello.search_backends.SEARCH_BACKEND_REGISTRY`.

    ``search_depth`` is the probe-time search depth (chunk 2's
    :func:`dialectical_othello.search.alphabeta` ply count). Default 4 is
    a reasonable Othello probe depth -- deep enough for a candidate to
    distinguish forced material swings, shallow enough that an
    end-to-end analyze on the starting position finishes in well under
    a second.

    ``deadline`` is a monotonic-clock cap (seconds; ``None`` for
    unlimited).
    """

    search_backend: str = "alphabeta"
    search_depth: int = 4
    deadline: float | None = None


@dataclass(frozen=True)
class OthelloEngineDecision:
    """The Othello engine's chosen move and the probe it came from.

    Mirrors the core :class:`dialectical_games.engine.EngineDecision`
    shape; carries ``move_id`` (the algebraic move string -- ``"d3"``,
    ``"e6"``, ``"pass"``) plus the originating :class:`OthelloMoveProbe`
    (or ``None`` if the position was terminal).
    """

    move_id: str
    selected: OthelloMoveProbe | None


@dataclass(frozen=True)
class OthelloEngineAnalysis:
    """Othello engine analysis -- the probes and the decision.

    Mirrors the core :class:`dialectical_games.engine.EngineAnalysis`
    surface; drops the core ``graph`` field (callers that need the
    graded layer can rebuild it from probes + a fresh policy). The
    Othello v1 chess-equivalent is a strictly thinner surface -- there
    is no post-decision hook so there is no need to inspect intermediate
    states.
    """

    probes: tuple[OthelloMoveProbe, ...]
    decision: OthelloEngineDecision


class OthelloEngine:
    """Othello engine -- implements the core :class:`Cartridge` Protocol.

    ``probe_moves(board)`` delegates to
    :func:`dialectical_othello.probe.probe_moves` (the chunk-3 driver
    that walks the WITNESS_RULES table per legal move and emits one
    :class:`OthelloMoveProbe` per move).

    ``make_graded_policy(board)`` returns a fresh
    :class:`OthelloGradedPolicy` bound to ``board`` (the bind is a
    no-op; the policy reads the survivor probes only, via
    :meth:`OthelloGradedPolicy.with_probes`).

    ``analyze(board)`` constructs the core orchestrator's settings
    carrier from this engine's :class:`OthelloEngineSettings` and calls
    :func:`dialectical_games.engine.analyze` with ``post_decision=None``
    -- the entire engine output is the core depth-0 pipeline. The
    chosen probe is unwrapped to an :class:`OthelloEngineDecision` and
    the probe tuple is narrowed back to the cartridge-typed
    :class:`OthelloMoveProbe` (every probe in the core return came from
    this engine's ``probe_moves`` so the cast is always sound).
    """

    def __init__(self, settings: OthelloEngineSettings | None = None) -> None:
        self.settings = settings or OthelloEngineSettings()

    # --- Cartridge Protocol ------------------------------------------------

    def probe_moves(self, board: Any) -> tuple[CoreMoveProbe, ...]:
        if not isinstance(board, OthelloBoard):
            raise TypeError(
                "OthelloEngine.probe_moves requires an OthelloBoard, "
                f"got {type(board).__name__}"
            )
        return probe_moves(board)

    def make_graded_policy(self, board: Any) -> OthelloGradedPolicy:
        return OthelloGradedPolicy(board=board)

    # --- Othello-facing driver --------------------------------------------

    def analyze(self, board: OthelloBoard) -> OthelloEngineAnalysis:
        """Run the depth-0 pipeline on ``board``.

        Threads the cartridge settings (the full
        :class:`OthelloEngineSettings`) onto
        ``CoreEngineSettings.cartridge_settings`` so a future
        post-decision hook can read its own knobs; forwards the
        load-bearing ``search_backend`` and ``deadline`` onto the core
        settings directly. ``post_decision`` is ``None`` per scout §5
        -- Othello v1 has no candidate-restricted refutation phase.
        """
        core_settings = CoreEngineSettings(
            search_backend=self.settings.search_backend,
            deadline=self.settings.deadline,
            cartridge_settings=self.settings,
        )
        analysis: CoreEngineAnalysis = core_analyze(
            board,
            cartridge=self,
            settings=core_settings,
            post_decision=None,
        )
        othello_probes = tuple(
            cast(OthelloMoveProbe, p) for p in analysis.probes
        )
        selected_probe: OthelloMoveProbe | None
        if analysis.decision.selected is None:
            selected_probe = None
        else:
            selected_probe = cast(OthelloMoveProbe, analysis.decision.selected)
        decision = OthelloEngineDecision(
            move_id=analysis.decision.move_id,
            selected=selected_probe,
        )
        return OthelloEngineAnalysis(probes=othello_probes, decision=decision)

    def choose_move(self, board: OthelloBoard) -> OthelloEngineDecision:
        """Return only the engine's decision (skip the probe tuple)."""
        return self.analyze(board).decision
