"""Typed evidence vocabulary for the Othello cartridge (plan §4, chunk 3).

Mirrors the chess post-``74fe4f0`` shape (``dialectical_chess/evidence.py``):
typed ``SupportKind`` / ``ObjectionKind`` enums + frozen dataclasses with a
typed ``support_magnitude`` / ``objection_magnitude: int | None`` field. No
string-prefix parsing anywhere — every dispatch keys on the enum.

Othello has no DefeaterKind / ReplyKind at v1 (plan §4 — no
``defense:heuristic_suppression`` channel use yet, and no post-decision hook
this chunk). Those are reserved for chunk 5+ if the strength gate demands
them.

The enum values are sourced directly from plan §4 (the 14-row HEURISTIC
table — 12 new, 2 reused). They are NOT invented beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from dialectical_games.scheme import Tier


# --- World classifier -------------------------------------------------------
#
# Othello has fewer "worlds" than chess (no SMT, no SEARCH refutation channel
# at v1). The PROCEDURAL world is the catch-all for positional reasoning;
# TERMINAL is the only FACT-tier world Othello cares about at v1.


class EvidenceWorld(str, Enum):
    POSITIONAL = "positional"
    MATERIAL = "material"
    TERMINAL = "terminal"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


# --- Role -------------------------------------------------------------------


class EvidenceRole(str, Enum):
    SUPPORT = "support"
    OBJECTION = "objection"


# --- Support kinds (plan §4) ------------------------------------------------
#
# One value per support row. ``GENERIC`` is the unclassified default (parallels
# chess's ``SupportKind.GENERIC``). ``TERMINAL_WIN`` is the FACT-tier support
# reused from core (plan §3); the rest are HEURISTIC.


class SupportKind(str, Enum):
    GENERIC = "generic"
    TERMINAL_WIN = "terminal_win"
    CORNER_OCCUPIED = "corner_occupied"
    STABLE_DISC = "stable_disc"
    MOBILITY_GAIN = "mobility_gain"
    FRONTIER_REDUCED = "frontier_reduced"
    EDGE_ANCHOR = "edge_anchor"
    CRAMPS_OPPONENT = "cramps_opponent"
    PARITY_HELD = "parity_held"
    MATERIAL_DISC_DIFF = "material_disc_diff"


# --- Objection kinds (plan §4) ----------------------------------------------


class ObjectionKind(str, Enum):
    NONE = "none"
    TERMINAL_LOSS = "terminal_loss"
    CORNER_CONCEDED = "corner_conceded"
    X_SQUARE_PLAYED = "x_square_played"
    C_SQUARE_PLAYED = "c_square_played"
    FRONTIER_EXPOSED = "frontier_exposed"
    STABILITY_LOSS_TO_OPPONENT = "stability_loss_to_opponent"
    PARITY_LOST = "parity_lost"
    MOBILITY_LOSS = "mobility_loss"
    MATERIAL_DISC_DIFF = "material_disc_diff"


# --- Dataclasses ------------------------------------------------------------
#
# Frozen, typed magnitudes. ``magnitude`` is ``None`` for BOOLEAN witnesses
# (corner / X-square / parity); an ``int`` for COUNT witnesses (mobility /
# frontier / stability / material).


@dataclass(frozen=True)
class SupportEvidence:
    """A typed support reason emitted by an Othello witness producer."""

    label: str
    world: EvidenceWorld
    support_kind: SupportKind = SupportKind.GENERIC
    support_magnitude: int | None = None
    role: EvidenceRole = EvidenceRole.SUPPORT
    tier: Tier = Tier.HEURISTIC


@dataclass(frozen=True)
class ObjectionEvidence:
    """A typed objection emitted by an Othello witness producer."""

    label: str
    world: EvidenceWorld
    objection_kind: ObjectionKind = ObjectionKind.NONE
    objection_magnitude: int | None = None
    role: EvidenceRole = EvidenceRole.OBJECTION
    tier: Tier = Tier.HEURISTIC


ArgumentEvidence: TypeAlias = SupportEvidence | ObjectionEvidence


# --- Tier classification (mirrors chess's ``objection_tier``) ---------------
#
# The FACT/HEURISTIC split is enum-keyed, not string-prefix-keyed. Only the
# terminal-outcome kinds are FACT at v1; everything else is positional
# judgement.


_FACT_SUPPORT_KINDS: frozenset[SupportKind] = frozenset({SupportKind.TERMINAL_WIN})
_FACT_OBJECTION_KINDS: frozenset[ObjectionKind] = frozenset(
    {ObjectionKind.TERMINAL_LOSS}
)


def support_tier(support_kind: SupportKind) -> Tier:
    """Return the :class:`Tier` of a support kind."""
    if support_kind in _FACT_SUPPORT_KINDS:
        return Tier.FACT
    return Tier.HEURISTIC


def objection_tier(objection_kind: ObjectionKind) -> Tier:
    """Return the :class:`Tier` of an objection kind."""
    if objection_kind in _FACT_OBJECTION_KINDS:
        return Tier.FACT
    return Tier.HEURISTIC


# --- Constructors -----------------------------------------------------------
#
# Tiny constructor functions parallel to chess's ``support_evidence`` /
# ``objection_evidence`` so the WITNESS_RULES table emits consistently-tiered
# evidence without each rule having to compute the tier itself.


def support_evidence(
    label: str,
    *,
    world: EvidenceWorld,
    support_kind: SupportKind = SupportKind.GENERIC,
    support_magnitude: int | None = None,
) -> SupportEvidence:
    """Build a :class:`SupportEvidence` with auto-classified tier."""
    return SupportEvidence(
        label=label,
        world=world,
        support_kind=support_kind,
        support_magnitude=support_magnitude,
        tier=support_tier(support_kind),
    )


def objection_evidence(
    label: str,
    *,
    world: EvidenceWorld,
    objection_kind: ObjectionKind,
    objection_magnitude: int | None = None,
) -> ObjectionEvidence:
    """Build an :class:`ObjectionEvidence` with auto-classified tier."""
    return ObjectionEvidence(
        label=label,
        world=world,
        objection_kind=objection_kind,
        objection_magnitude=objection_magnitude,
        tier=objection_tier(objection_kind),
    )
