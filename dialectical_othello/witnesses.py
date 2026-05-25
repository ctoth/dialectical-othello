"""Declarative ``WITNESS_RULES`` table for Othello (chunk 3 — plan §4, §6).

Each rule is a pure per-board predicate: given the parent board (side-to-move
is the mover) and the candidate move, it returns either ``None`` (rule does
not fire) or a typed :class:`ArgumentEvidence`. The probe driver
(:mod:`dialectical_othello.probe`) walks this table for every legal move and
accumulates the typed evidence into the per-move ``reasons`` / ``objections``
fields.

This mirrors the checkers ``WITNESS_RULES`` pattern
(``dialectical_checkers/witnesses.py:1000-1158``) — every Othello positional
witness is a deterministic per-board predicate, so the declarative table is
cleaner than chess's per-theme imperative producers.

No tuned constants: a rule either fires structurally (a corner is occupied —
binary), reports a magnitude (the frontier-disc count delta — numeric), or
does not fire. There are no firing thresholds of the form "rule fires only
when N > k".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dialectical_othello.board import (
    BOARD_MASK,
    OthelloBoard,
    OthelloMove,
    _SHIFTS,
    _algebraic,
)
from dialectical_othello.evidence import (
    ArgumentEvidence,
    EvidenceRole,
    EvidenceWorld,
    ObjectionKind,
    SupportKind,
    objection_evidence,
    support_evidence,
)


# --- Square sets (plan §4) --------------------------------------------------
#
# Algebraic squares for the four corners and their immediate diagonal /
# orthogonal neighbours. Square indices use ``file + 8*rank`` (board.py:14).
# Computed by name rather than literal so the reader can verify against the
# plan §4 enumeration without doing arithmetic.


def _sq(name: str) -> int:
    """Return the 0..63 bit index for an algebraic name like ``"a1"``."""
    file_ch, rank_ch = name[0], name[1]
    return "abcdefgh".index(file_ch) + 8 * (int(rank_ch) - 1)


CORNERS: frozenset[int] = frozenset({_sq("a1"), _sq("h1"), _sq("a8"), _sq("h8")})
X_SQUARES: frozenset[int] = frozenset(
    {_sq("b2"), _sq("g2"), _sq("b7"), _sq("g7")}
)
C_SQUARES: frozenset[int] = frozenset(
    {
        _sq("b1"), _sq("g1"),
        _sq("a2"), _sq("h2"),
        _sq("a7"), _sq("h7"),
        _sq("b8"), _sq("g8"),
    }
)


def _empty_bb(board: OthelloBoard) -> int:
    """Bitboard of empty squares on ``board``."""
    return BOARD_MASK ^ (board.black | board.white)


def _own_bb(board: OthelloBoard, color: str) -> int:
    """Bitboard of discs of ``color`` (``"B"`` or ``"W"``) on ``board``."""
    if color == "B":
        return board.black
    return board.white


def _frontier_count(board: OthelloBoard, color: str) -> int:
    """Count of ``color``'s frontier discs (own discs adjacent to >=1 empty).

    Standard Othello "frontier" = a disc with at least one of its 8 neighbour
    squares empty. Computed by union-shifting the empty bitboard in all 8
    directions; any own-disc intersecting that union is a frontier disc.
    """
    empty = _empty_bb(board)
    neighbours_of_empty = 0
    for shift in _SHIFTS:
        neighbours_of_empty |= shift(empty)
    own = _own_bb(board, color)
    return (own & neighbours_of_empty).bit_count()


def _legal_move_count_for(board: OthelloBoard, color: str) -> int:
    """Count of ``color``'s legal disc-flips on ``board``.

    The board's ``legal_moves()`` reports for the side-to-move only; this
    helper computes the same quantity for an arbitrary colour by temporarily
    re-tagging the board's turn. The bitboard payload is unchanged so the
    result is invariant under turn re-tagging.
    """
    if board.turn == color:
        bb = board._legal_moves_bb()
    else:
        bb = OthelloBoard(
            black=board.black,
            white=board.white,
            turn=color,
            pass_count=board.pass_count,
        )._legal_moves_bb()
    return bb.bit_count()


# --- Edge-anchor helper -----------------------------------------------------
#
# Per plan §4 #9: an edge-adjacent square is "anchored" iff a same-colour
# corner already exists on the same edge AND all squares strictly between
# the corner and the move's square (along that edge) are same-colour discs
# on the resulting board. This is the cheap, structural form of stability
# the plan endorses for chunk 3.

#: For each edge square, the (corner_square, between_squares) tuples it could
#: anchor against — at most one per edge. ``between_squares`` is the list of
#: squares strictly between ``corner_square`` and the edge square inclusive
#: of NEITHER endpoint. An empty between-list means the edge square is
#: immediately adjacent to the corner (a C-square — anchoring is automatic
#: if the corner is same-coloured).
def _edge_anchor_targets() -> dict[int, tuple[tuple[int, tuple[int, ...]], ...]]:
    edges_with_corners: dict[int, tuple[tuple[int, tuple[int, ...]], ...]] = {}
    edge_specs = (
        ("a", (1, 2, 3, 4, 5, 6, 7, 8)),  # file a, ranks 1..8
        ("h", (1, 2, 3, 4, 5, 6, 7, 8)),  # file h
    )
    for file_ch, ranks in edge_specs:
        squares_on_edge = tuple(_sq(f"{file_ch}{r}") for r in ranks)
        corner_low = squares_on_edge[0]
        corner_high = squares_on_edge[-1]
        for idx, sq in enumerate(squares_on_edge):
            if sq in CORNERS:
                continue
            below = squares_on_edge[1:idx]
            above = squares_on_edge[idx + 1:-1]
            edges_with_corners[sq] = (
                (corner_low, below),
                (corner_high, above),
            )
    # Ranks 1 and 8 — rows.
    for rank in (1, 8):
        squares_on_edge = tuple(_sq(f"{f}{rank}") for f in "abcdefgh")
        corner_low = squares_on_edge[0]
        corner_high = squares_on_edge[-1]
        for idx, sq in enumerate(squares_on_edge):
            if sq in CORNERS:
                continue
            below = squares_on_edge[1:idx]
            above = squares_on_edge[idx + 1:-1]
            edges_with_corners[sq] = (
                (corner_low, below),
                (corner_high, above),
            )
    return edges_with_corners


_EDGE_ANCHOR_TARGETS = _edge_anchor_targets()


def _is_edge_anchor(board_after: OthelloBoard, color: str, square: int) -> bool:
    """True iff ``square`` on ``board_after`` is part of an unbroken
    same-colour edge run anchored on a same-colour corner."""
    targets = _EDGE_ANCHOR_TARGETS.get(square)
    if targets is None:
        return False
    own = _own_bb(board_after, color)
    if not (own & (1 << square)):
        # The mover's own disc must occupy ``square`` post-move (it should,
        # since ``square`` is the placement, but the rule is defensive).
        return False
    for corner_sq, between in targets:
        if not (own & (1 << corner_sq)):
            continue
        if all(own & (1 << s) for s in between):
            return True
    return False


# --- Predicate helpers ------------------------------------------------------
#
# Each WITNESS_RULES entry's ``predicate`` returns ``None`` if the rule does
# not fire, else a typed :class:`ArgumentEvidence`. Predicates take
# ``(board, move)`` plus the precomputed ``child`` (the result of
# ``board.apply(move)``) so each rule does not re-apply.


def _pro_corner_occupied(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, child
    sq = move.square
    if sq is None or sq not in CORNERS:
        return None
    return support_evidence(
        f"pro:corner:occupied:{_algebraic(sq)}",
        world=EvidenceWorld.POSITIONAL,
        support_kind=SupportKind.CORNER_OCCUPIED,
    )


def _obj_corner_conceded(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, move
    # The opponent (= child.turn) can land on a corner next turn.
    for opp_move in child.legal_moves():
        if opp_move.square is not None and opp_move.square in CORNERS:
            return objection_evidence(
                "obj:corner:conceded",
                world=EvidenceWorld.POSITIONAL,
                objection_kind=ObjectionKind.CORNER_CONCEDED,
            )
    return None


def _obj_x_square_played(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, child
    sq = move.square
    if sq is None or sq not in X_SQUARES:
        return None
    return objection_evidence(
        f"obj:x_square:played:{_algebraic(sq)}",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.X_SQUARE_PLAYED,
    )


def _obj_c_square_played(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, child
    sq = move.square
    if sq is None or sq not in C_SQUARES:
        return None
    return objection_evidence(
        f"obj:c_square:played:{_algebraic(sq)}",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.C_SQUARE_PLAYED,
    )


def _pro_mobility_gain(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del move
    mover_color = board.turn
    before = _legal_move_count_for(board, mover_color)
    after = _legal_move_count_for(child, mover_color)
    delta = after - before
    if delta <= 0:
        return None
    return support_evidence(
        f"pro:mobility:{delta}",
        world=EvidenceWorld.POSITIONAL,
        support_kind=SupportKind.MOBILITY_GAIN,
        support_magnitude=delta,
    )


def _pro_cramps_opponent(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del move
    opp_color = "W" if board.turn == "B" else "B"
    before = _legal_move_count_for(board, opp_color)
    after = _legal_move_count_for(child, opp_color)
    drop = before - after
    if drop <= 0:
        return None
    return support_evidence(
        f"pro:cramps_opponent:{drop}",
        world=EvidenceWorld.POSITIONAL,
        support_kind=SupportKind.CRAMPS_OPPONENT,
        support_magnitude=drop,
    )


def _pro_frontier_reduced(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del move
    mover_color = board.turn
    before = _frontier_count(board, mover_color)
    after = _frontier_count(child, mover_color)
    drop = before - after
    if drop <= 0:
        return None
    return support_evidence(
        f"pro:frontier:reduced:{drop}",
        world=EvidenceWorld.POSITIONAL,
        support_kind=SupportKind.FRONTIER_REDUCED,
        support_magnitude=drop,
    )


def _obj_frontier_exposed(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del move
    mover_color = board.turn
    before = _frontier_count(board, mover_color)
    after = _frontier_count(child, mover_color)
    growth = after - before
    if growth <= 0:
        return None
    return objection_evidence(
        f"obj:frontier:{growth}",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.FRONTIER_EXPOSED,
        objection_magnitude=growth,
    )


def _pro_edge_anchor(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    sq = move.square
    if sq is None:
        return None
    if sq in CORNERS:
        # Corner-on-corner is already a stronger ``pro:corner:occupied``; do
        # not double-fire the edge anchor on the corner itself.
        return None
    if sq not in _EDGE_ANCHOR_TARGETS:
        return None
    mover_color = board.turn
    if not _is_edge_anchor(child, mover_color, sq):
        return None
    return support_evidence(
        f"pro:edge:anchor:{_algebraic(sq)}",
        world=EvidenceWorld.POSITIONAL,
        support_kind=SupportKind.EDGE_ANCHOR,
    )


# --- STUBS — return None this chunk (plan §8 chunk 3, D3 stub policy) -----
#
# Stability and parity are deferred to chunk 5 (or a v2 follow-up). The
# WITNESS_RULES table still carries entries so the producer surface is
# explicit: each enum has a registered rule, the rule simply never fires
# yet. Adding the real predicate later is a pure substitution of the
# ``label`` callback — the rest of the pipeline already accepts the
# corresponding typed evidence + core label.


def _pro_stability_stub(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, move, child
    return None


def _obj_stability_loss_stub(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, move, child
    return None


def _pro_parity_stub(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, move, child
    return None


def _obj_parity_stub(
    board: OthelloBoard, move: OthelloMove, child: OthelloBoard
) -> ArgumentEvidence | None:
    del board, move, child
    return None


# --- WITNESS_RULES table ----------------------------------------------------


WitnessPredicate = Callable[
    [OthelloBoard, OthelloMove, OthelloBoard], "ArgumentEvidence | None"
]


@dataclass(frozen=True)
class WitnessRule:
    """One row of the Othello witness producer table (plan §6).

    ``name`` is the typed-evidence kind for diagnostics. ``role`` is whether
    the rule emits :data:`EvidenceRole.SUPPORT` or
    :data:`EvidenceRole.OBJECTION` — the probe driver uses this to route the
    rule's output into ``reasons`` or ``objections``. ``predicate`` is the
    per-board firing predicate.
    """

    name: str
    role: EvidenceRole
    predicate: WitnessPredicate
    stub: bool = False


WITNESS_RULES: tuple[WitnessRule, ...] = (
    # --- BOOLEAN positional witnesses (corners + edge) ---------------------
    WitnessRule(
        name="CORNER_OCCUPIED",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_corner_occupied,
    ),
    WitnessRule(
        name="CORNER_CONCEDED",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_corner_conceded,
    ),
    WitnessRule(
        name="X_SQUARE_PLAYED",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_x_square_played,
    ),
    WitnessRule(
        name="C_SQUARE_PLAYED",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_c_square_played,
    ),
    WitnessRule(
        name="EDGE_ANCHOR",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_edge_anchor,
    ),
    # --- COUNT positional witnesses (mobility / frontier) ------------------
    WitnessRule(
        name="MOBILITY_GAIN",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_mobility_gain,
    ),
    WitnessRule(
        name="CRAMPS_OPPONENT",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_cramps_opponent,
    ),
    WitnessRule(
        name="FRONTIER_REDUCED",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_frontier_reduced,
    ),
    WitnessRule(
        name="FRONTIER_EXPOSED",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_frontier_exposed,
    ),
    # --- STUBS — never fire this chunk (plan §8 D3) ------------------------
    WitnessRule(
        name="STABILITY_GAIN",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_stability_stub,
        stub=True,
    ),
    WitnessRule(
        name="STABILITY_LOSS_TO_OPPONENT",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_stability_loss_stub,
        stub=True,
    ),
    WitnessRule(
        name="PARITY_HELD",
        role=EvidenceRole.SUPPORT,
        predicate=_pro_parity_stub,
        stub=True,
    ),
    WitnessRule(
        name="PARITY_LOST",
        role=EvidenceRole.OBJECTION,
        predicate=_obj_parity_stub,
        stub=True,
    ),
)
