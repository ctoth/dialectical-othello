"""Probe-construction driver for the Othello cartridge (chunk 3 — plan §6, §8).

:func:`probe_moves` is the per-position fan-out: it walks
:data:`WITNESS_RULES` for each legal move, accumulates typed evidence into
the cartridge-extension fields, and populates the core ``reasons`` /
``objections`` fields with the translated core-taxonomy labels via
:mod:`dialectical_othello.core_labels`.

The probe shape extends :class:`dialectical_games.arguments.MoveProbe` with
two cartridge-side fields — ``reason_evidence`` and ``objection_evidence``
— carrying the typed :class:`ArgumentEvidence` tuples. Parallel to chess's
``ChessMoveProbe`` (``dialectical_chess/arguments.py`` pattern). Chunk 4
finalises the GradedPolicy that consumes these extension fields; chunk 3
just lays the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dialectical_games.arguments import MoveProbe

from dialectical_games.scheme import Tier

from dialectical_othello.board import OthelloBoard, OthelloMove
from dialectical_othello.core_labels import core_labels_for_probe
from dialectical_othello.evidence import (
    ArgumentEvidence,
    EvidenceRole,
)
from dialectical_othello.search import static_evaluation
from dialectical_othello.witnesses import WITNESS_RULES


@dataclass(frozen=True)
class OthelloMoveProbe(MoveProbe):
    """An Othello move probe.

    Inherits the core ``MoveProbe`` fields (``move_id``, ``reasons``,
    ``objections``, ``child_eval``, ``contested``, ...) and adds the
    cartridge-extension fields: the typed :class:`OthelloMove` so the
    search backend reads the move directly (no algebraic round-trip),
    and the typed :class:`ArgumentEvidence` tuples the graded policy
    consumes for per-position CDF construction. The core graph builder
    reads the translated ``reasons`` / ``objections`` strings on the
    inherited fields.
    """

    move: OthelloMove = field(default_factory=OthelloMove.pass_move)
    reason_evidence: tuple[ArgumentEvidence, ...] = field(default_factory=tuple)
    objection_evidence: tuple[ArgumentEvidence, ...] = field(default_factory=tuple)


def _evaluate_for_parent_mover(parent: OthelloBoard, child: OthelloBoard) -> int:
    """Return the parent-mover's view of ``child``'s static evaluation.

    :func:`static_evaluation` is mover-relative for the board it's evaluating
    — on the child the mover has flipped, so the raw value is the opponent's
    view. The parent-mover's view is the negation. Terminal children carry
    the same convention.
    """
    raw = static_evaluation(child)
    if child.turn == parent.turn:
        # Auto-pass case: the parent's mover still has the move on the child
        # (the opponent passed). Same mover -> same sign.
        return raw
    return -raw


def _build_probe_for_move(parent: OthelloBoard, move) -> OthelloMoveProbe:  # type: ignore[no-untyped-def]
    """Compute typed evidence + core labels for one legal move."""
    child = parent.apply(move)
    reasons_typed: list[ArgumentEvidence] = []
    objections_typed: list[ArgumentEvidence] = []
    for rule in WITNESS_RULES:
        result = rule.predicate(parent, move, child)
        if result is None:
            continue
        if rule.role is EvidenceRole.SUPPORT:
            reasons_typed.append(result)
        else:
            objections_typed.append(result)
    reasons_t = tuple(reasons_typed)
    objections_t = tuple(objections_typed)
    core_reasons, core_objections = core_labels_for_probe(
        reason_evidence=reasons_t,
        objection_evidence=objections_t,
    )
    contested = _has_heuristic(reasons_t) and _has_heuristic(objections_t)
    return OthelloMoveProbe(
        move_id=move.move_id(),
        reasons=core_reasons,
        objections=core_objections,
        child_eval=_evaluate_for_parent_mover(parent, child),
        contested=contested,
        move=move,
        reason_evidence=reasons_t,
        objection_evidence=objections_t,
    )


def _has_heuristic(evidence_items: tuple[ArgumentEvidence, ...]) -> bool:
    """True iff the tuple contains at least one HEURISTIC-tier evidence."""
    for item in evidence_items:
        if item.tier is Tier.HEURISTIC:
            return True
    return False


def probe_moves(board: OthelloBoard) -> tuple[OthelloMoveProbe, ...]:
    """Return one :class:`OthelloMoveProbe` per legal move on ``board``.

    Order matches :meth:`OthelloBoard.legal_moves`. Terminal boards return
    an empty tuple. The probe driver does not call into the graded policy
    — that's chunk 4 — but the typed evidence and core labels it populates
    are exactly what the graded policy will read.
    """
    moves = board.legal_moves()
    return tuple(_build_probe_for_move(board, m) for m in moves)
