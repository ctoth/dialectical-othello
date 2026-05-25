"""Othello cartridge: :class:`OthelloGradedPolicy` (H' principled derivation).

Implements the core :class:`dialectical_games.arguments.GradedPolicy` Protocol
for the Othello cartridge. A per-build policy bound to the survivor probe set;
the bound policy carries one per-position cache:

* a per-position rank-fraction over sibling ``child_eval`` values (the move
  base rate), built from the survivor probes.

The cache is populated in :meth:`OthelloGradedPolicy.with_probes`, which the
generic builder calls once at entry (chunk H'.a Protocol extension).

Unlike the chess cartridge, Othello has **no MATERIAL-class HEURISTIC
witnesses** at v1 (plan §3): every disc is identical, the disc-count
differential is a TERMINAL signal (translated to ``pro:terminal_win`` /
``obj:terminal_loss``) rather than a per-probe HEURISTIC. The
:meth:`with_probes` cache therefore only carries the ``child_eval`` CDF —
no per-prefix MATERIAL CDF is needed.

Witness opinions classify the label as BOOLEAN (a FIXED HEURISTIC label
with no magnitude) or COUNT (a magnitude-carrying HEURISTIC label) by
membership in the chunk-3 dispatch dicts; both dispatch to
``Opinion.from_evidence(n, 0, MAX_ENT_PRIOR)`` -- the beta-binomial
conjugate prior. BOOLEAN is the ``n = 1`` case; COUNT carries
``n = magnitude``. A label that the chunk-3 dispatch dicts cannot classify
is a typing failure -- the typed pipeline is exhaustive -- and raises
:class:`ValueError` rather than falling back to a vacuous default.

The ONE literal that survives in the witness/policy path is
:data:`MAX_ENT_PRIOR` (= 0.5, the max-entropy binary prior); every other
number flows out of :func:`doxa.Opinion.from_evidence` or a per-position
Hazen rank-fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from doxa import Opinion

from dialectical_games.arguments import GradedPolicy, MoveProbe

from dialectical_othello.core_labels import (
    _FIXED_OBJECTION_BY_KIND,
    _FIXED_SUPPORT_BY_KIND,
    _MAGNITUDE_OBJECTION_PREFIX_BY_KIND,
    _MAGNITUDE_SUPPORT_PREFIX_BY_KIND,
)


__all__ = [
    "MAX_ENT_PRIOR",
    "OthelloGradedPolicy",
    "make_graded_policy",
]


#: The max-entropy binary prior. The ONLY literal that survives in the
#: witness / policy path -- every other number is derived from witness
#: semantics through :func:`doxa.Opinion.from_evidence` or from a
#: per-position Hazen rank-fraction.
MAX_ENT_PRIOR: float = 0.5


# --- Label classification tables (built from chunk-3 dispatch dicts) -------
#
# The chunk-3 translator (:mod:`dialectical_othello.core_labels`) is the
# authoritative typed-evidence -> core-label boundary. The graded policy
# inverts the translator's dispatch dicts so every core label it sees is
# classified by the same source-of-truth: a label in the FIXED values is
# BOOLEAN; a label whose prefix is in the MAGNITUDE values is COUNT;
# anything else raises.

_BOOLEAN_LABELS: frozenset[str] = frozenset(
    set(_FIXED_SUPPORT_BY_KIND.values()) | set(_FIXED_OBJECTION_BY_KIND.values())
)

_COUNT_PREFIXES: frozenset[str] = frozenset(
    set(_MAGNITUDE_SUPPORT_PREFIX_BY_KIND.values())
    | set(_MAGNITUDE_OBJECTION_PREFIX_BY_KIND.values())
)


class _WitnessClass(Enum):
    """Two-class witness taxonomy for Othello (no MATERIAL — plan §3)."""

    COUNT = auto()
    BOOLEAN = auto()


def _label_prefix(label: str) -> str:
    """Return the label's magnitude-bearing prefix (everything before ``:{n}``).

    A magnitude-carrying label is ``<prefix>:<n>``;
    ``label.rpartition(":")[0]`` returns ``<prefix>``. Symmetric to the
    chess cartridge.
    """
    return label.rpartition(":")[0]


def _classify(label: str, magnitude: int | None) -> _WitnessClass:
    """Classify ``(label, magnitude)`` against the chunk-3 dispatch dicts.

    A label with no magnitude must be in :data:`_BOOLEAN_LABELS` (the
    union of the translator's FIXED dispatch values) -> BOOLEAN.
    A magnitude-carrying label's prefix must be in :data:`_COUNT_PREFIXES`
    (the union of the translator's MAGNITUDE prefix dispatch values) ->
    COUNT.

    The typed pipeline is exhaustive: every label that reaches the policy
    was produced by the chunk-3 translator, which only ever emits labels
    keyed in those dispatch dicts. An unknown label is a typing failure
    and raises :class:`ValueError` rather than silently returning a
    vacuous default opinion.
    """
    if magnitude is None:
        if label in _BOOLEAN_LABELS:
            return _WitnessClass.BOOLEAN
        raise ValueError(
            f"OthelloGradedPolicy: unknown BOOLEAN witness label {label!r}"
        )
    prefix = _label_prefix(label)
    if prefix in _COUNT_PREFIXES:
        return _WitnessClass.COUNT
    raise ValueError(
        f"OthelloGradedPolicy: unknown COUNT witness label {label!r} "
        f"(prefix {prefix!r} not in dispatch table)"
    )


@dataclass(frozen=True)
class OthelloGradedPolicy:
    """``GradedPolicy`` implementation for the Othello cartridge (chunk H').

    The bound root ``board`` is retained on the dataclass for backward
    compatibility with the orchestrator's
    ``Cartridge.make_graded_policy(board)`` signature, but is not read.
    The per-position cache (``_child_eval_ranks``) is populated only by
    :meth:`with_probes`; an unbound policy (no :meth:`with_probes` call
    yet) falls back to the neutral max-entropy prior.
    """

    board: Any = None
    _child_eval_ranks: dict[str, float] = field(default_factory=dict)
    _child_eval_count: int = 0

    def with_probes(
        self, probes: Sequence[MoveProbe]
    ) -> "OthelloGradedPolicy":
        """Return a policy bound to ``probes`` (chunk H' D1, D4).

        Builds the per-position ``child_eval`` rank-fraction cache.
        Othello ``child_eval`` is **mover-relative** (the probe driver
        stores the parent-mover's view of the child static evaluation,
        :func:`dialectical_othello.probe._evaluate_for_parent_mover`):
        LARGER is better for the mover, so the largest gets the highest
        rank-fraction -- ASCENDING orientation, same as chess. Mid-rank
        averaging handles ties so equal-``child_eval`` probes share the
        same rank-fraction.

        No MATERIAL CDF: Othello has no MATERIAL-class HEURISTIC witness
        at v1 (plan §3); the disc-count differential is a TERMINAL
        signal, not a per-probe HEURISTIC. Returns ``self``'s board
        binding on the new policy so chained binds preserve it.
        """
        n = len(probes)
        if n == 0:
            return OthelloGradedPolicy(board=self.board)

        sorted_probes = sorted(probes, key=lambda p: p.child_eval)
        ranks: dict[str, float] = {}
        i = 0
        while i < n:
            j = i + 1
            while (
                j < n
                and sorted_probes[j].child_eval == sorted_probes[i].child_eval
            ):
                j += 1
            mid = ((i + 1) + j) / 2.0
            frac = mid / (n + 1)
            for k in range(i, j):
                ranks[sorted_probes[k].move_id] = frac
            i = j

        return OthelloGradedPolicy(
            board=self.board,
            _child_eval_ranks=ranks,
            _child_eval_count=n,
        )

    def move_base_rate(self, probe: MoveProbe) -> float:
        """The move node's base rate ``a`` -- per-position rank-fraction.

        Reads the per-position CDF :meth:`with_probes` built. Othello
        ``child_eval`` is mover-relative; larger is better for the mover;
        the largest ``child_eval`` gets the highest rank-fraction. With
        an unbound policy (no :meth:`with_probes` call yet) the base
        rate is the neutral max-entropy prior :data:`MAX_ENT_PRIOR`.
        """
        rank_fraction = self._child_eval_ranks.get(probe.move_id)
        if rank_fraction is None:
            return MAX_ENT_PRIOR
        return rank_fraction

    def witness_opinion(
        self,
        *,
        probe: MoveProbe,
        label: str,
        magnitude: int | None,
    ) -> Opinion:
        """A HEURISTIC witness opinion (chunk H' D3).

        Classifies ``(label, magnitude)`` against the chunk-3 dispatch
        dicts and dispatches:

        * BOOLEAN -> ``Opinion.from_evidence(1, 0, MAX_ENT_PRIOR)`` --
          a single observation: "the witness fired."
        * COUNT   -> ``Opinion.from_evidence(magnitude, 0, MAX_ENT_PRIOR)``
          -- each unit of the count is one observation.

        Both derive from the beta-binomial conjugate prior. There is no
        MATERIAL branch (Othello has no MATERIAL-class HEURISTIC at v1
        -- plan §3). An unknown label raises :class:`ValueError`.
        """
        del probe  # classification is by label alone for Othello
        cls = _classify(label, magnitude)
        if cls is _WitnessClass.BOOLEAN:
            return Opinion.from_evidence(1.0, 0.0, MAX_ENT_PRIOR)
        assert cls is _WitnessClass.COUNT  # exhaustive dispatch
        assert magnitude is not None  # classify guarantees it
        return Opinion.from_evidence(float(magnitude), 0.0, MAX_ENT_PRIOR)

    @property
    def edge_trust(self) -> Opinion:
        """The (witness -> move) edge trust opinion.

        Edges are facts of the graph, not measured beliefs; structural
        trust is dogmatic-true. The base rate is the max-entropy binary
        prior :data:`MAX_ENT_PRIOR`.
        """
        return Opinion.dogmatic_true(MAX_ENT_PRIOR)


def make_graded_policy(board: Any = None) -> OthelloGradedPolicy:
    """Construct a per-build Othello graded policy bound to ``board``.

    Under chunk H' the constructor does not cache board-derived position
    features -- the per-position aggregate (the survivor ``child_eval``
    rank-fractions) is built in :meth:`OthelloGradedPolicy.with_probes`,
    which the generic builder calls once at entry. ``board`` is retained
    for call-site backward compatibility but is not read.
    """
    return OthelloGradedPolicy(board=board)


# Static Protocol conformance check -- this assignment is a no-op at
# runtime but pyright flags any drift from the GradedPolicy Protocol.
_PROTOCOL_CHECK: GradedPolicy = OthelloGradedPolicy()
