"""Translate Othello-typed witnesses to core-taxonomy label strings.

Mirrors ``dialectical_chess.core_labels`` shape (enum-keyed dispatch dicts,
NO ``startswith`` anywhere). The Othello cartridge emits typed
:class:`SupportEvidence` / :class:`ObjectionEvidence`; the core graph builder
reads ``pro:``/``obj:`` string labels from its closed taxonomy. This module
is the boundary.

At chunk 3 the new HEURISTIC rows are not yet in core taxonomy (chunk 5
adds them); the translator emits the labels anyway. Probes that surface
unknown labels to the core layer are filtered by the probe builder at the
core-label edge — :func:`core_labels_for_probe` only includes labels for
which a dispatch entry exists, so adding the row in chunk 5 is purely a
core-side enabling step.
"""

from __future__ import annotations

from dialectical_othello.evidence import (
    ArgumentEvidence,
    ObjectionEvidence,
    ObjectionKind,
    SupportEvidence,
    SupportKind,
)


# Fixed (BOOLEAN) support kinds — no magnitude suffix.
_FIXED_SUPPORT_BY_KIND: dict[SupportKind, str] = {
    SupportKind.TERMINAL_WIN: "pro:terminal_win",
    SupportKind.CORNER_OCCUPIED: "pro:corner:occupied",
    SupportKind.EDGE_ANCHOR: "pro:edge:anchor",
    SupportKind.PARITY_HELD: "pro:parity:holds",
}

# Magnitude-bearing (COUNT / MATERIAL) support kinds — appended ``:{n}``.
_MAGNITUDE_SUPPORT_PREFIX_BY_KIND: dict[SupportKind, str] = {
    SupportKind.MOBILITY_GAIN: "pro:mobility",
    SupportKind.CRAMPS_OPPONENT: "pro:cramps_opponent",
    SupportKind.FRONTIER_REDUCED: "pro:frontier:reduced",
    SupportKind.STABLE_DISC: "pro:stability",
    SupportKind.MATERIAL_DISC_DIFF: "pro:material:disc_diff",
}

# Fixed (BOOLEAN) objection kinds — no magnitude suffix.
_FIXED_OBJECTION_BY_KIND: dict[ObjectionKind, str] = {
    ObjectionKind.TERMINAL_LOSS: "obj:terminal_loss",
    ObjectionKind.CORNER_CONCEDED: "obj:corner:conceded",
    ObjectionKind.X_SQUARE_PLAYED: "obj:x_square:played",
    ObjectionKind.C_SQUARE_PLAYED: "obj:c_square:played",
    ObjectionKind.PARITY_LOST: "obj:parity:lost",
}

# Magnitude-bearing (COUNT / MATERIAL) objection kinds.
_MAGNITUDE_OBJECTION_PREFIX_BY_KIND: dict[ObjectionKind, str] = {
    ObjectionKind.FRONTIER_EXPOSED: "obj:frontier",
    ObjectionKind.STABILITY_LOSS_TO_OPPONENT: "obj:stability:opponent",
    ObjectionKind.MOBILITY_LOSS: "obj:mobility",
    ObjectionKind.MATERIAL_DISC_DIFF: "obj:material:disc_diff",
}


def core_reason_label(evidence: ArgumentEvidence) -> str | None:
    """Map a typed Othello support to a core-taxonomy reason label."""
    if not isinstance(evidence, SupportEvidence):
        return None
    fixed = _FIXED_SUPPORT_BY_KIND.get(evidence.support_kind)
    if fixed is not None:
        return fixed
    prefix = _MAGNITUDE_SUPPORT_PREFIX_BY_KIND.get(evidence.support_kind)
    if prefix is not None:
        magnitude = evidence.support_magnitude
        if magnitude is not None and magnitude > 0:
            return f"{prefix}:{magnitude}"
    return None


def core_objection_label(evidence: ArgumentEvidence) -> str | None:
    """Map a typed Othello objection to a core-taxonomy objection label."""
    if not isinstance(evidence, ObjectionEvidence):
        return None
    fixed = _FIXED_OBJECTION_BY_KIND.get(evidence.objection_kind)
    if fixed is not None:
        return fixed
    prefix = _MAGNITUDE_OBJECTION_PREFIX_BY_KIND.get(evidence.objection_kind)
    if prefix is not None:
        magnitude = evidence.objection_magnitude
        if magnitude is not None and magnitude > 0:
            return f"{prefix}:{magnitude}"
    return None


def core_labels_for_probe(
    *,
    reason_evidence: tuple[ArgumentEvidence, ...],
    objection_evidence: tuple[ArgumentEvidence, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(reasons, objections)`` core-taxonomy label tuples.

    Duplicates are suppressed so the core graph builder sees each label at
    most once per probe (parallels chess ``core_labels_for_probe`` at
    ``dialectical_chess/core_labels.py``).
    """
    reasons: list[str] = []
    for evidence in reason_evidence:
        label = core_reason_label(evidence)
        if label is not None and label not in reasons:
            reasons.append(label)
    objections: list[str] = []
    for evidence in objection_evidence:
        label = core_objection_label(evidence)
        if label is not None and label not in objections:
            objections.append(label)
    return tuple(reasons), tuple(objections)
