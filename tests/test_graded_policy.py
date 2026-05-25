"""Property tests for :class:`OthelloGradedPolicy` (chunk 4).

Verifies the H' principled-derivation invariants:

* opinion components in ``[0, 1]`` and ``b + d + u == 1``;
* COUNT-class opinion uncertainty monotone-decreasing in ``n``;
* ``with_probes`` returns a fresh policy (Protocol immutability);
* ``with_probes`` idempotence on the same probes;
* ``edge_trust`` is :func:`Opinion.dogmatic_true(MAX_ENT_PRIOR)`;
* the constants-audit invariant: ``MAX_ENT_PRIOR = 0.5`` is the only
  float literal in :mod:`dialectical_othello.graded_policy`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from doxa import Opinion

from dialectical_games.arguments import MoveProbe

from dialectical_othello.graded_policy import (
    MAX_ENT_PRIOR,
    OthelloGradedPolicy,
)


# --- Helpers ---------------------------------------------------------------

# The full set of HEURISTIC labels chunk 3 may emit (FACT-tier
# ``pro:terminal_win`` / ``obj:terminal_loss`` never reach the graded
# layer; they are filtered upstream by ``_heuristic_evidence``). The
# BOOLEAN list contains the FIXED label values for every HEURISTIC
# ``SupportKind`` / ``ObjectionKind`` registered in
# :mod:`dialectical_othello.core_labels`.

_BOOLEAN_HEURISTIC_LABELS: tuple[str, ...] = (
    "pro:corner:occupied",
    "pro:edge:anchor",
    "pro:parity:holds",
    "obj:corner:conceded",
    "obj:x_square:played",
    "obj:c_square:played",
    "obj:parity:lost",
)

_COUNT_PREFIXES: tuple[str, ...] = (
    "pro:mobility",
    "pro:cramps_opponent",
    "pro:frontier:reduced",
    "pro:stability",
    "pro:material:disc_diff",
    "obj:frontier",
    "obj:stability:opponent",
    "obj:mobility",
    "obj:material:disc_diff",
)


def _probe(move_id: str, child_eval: int = 0) -> MoveProbe:
    """Construct a minimal :class:`MoveProbe` for the policy tests.

    The policy reads only ``move_id`` and ``child_eval`` from the probe
    (the witness-evidence fields are inspected by the graph builder, not
    the policy itself). Using the core :class:`MoveProbe` rather than
    :class:`OthelloMoveProbe` keeps the tests focused on the policy's
    Protocol surface.
    """
    return MoveProbe(move_id=move_id, child_eval=child_eval)


def _assert_opinion_well_formed(opinion: Opinion) -> None:
    """Assert each component lies in ``[0, 1]`` and ``b + d + u == 1``."""
    for component in (opinion.b, opinion.d, opinion.u, opinion.a):
        assert 0.0 <= component <= 1.0
    # Jøsang invariant: b + d + u == 1 (within floating tolerance).
    assert abs((opinion.b + opinion.d + opinion.u) - 1.0) < 1e-9


# --- Property tests --------------------------------------------------------


@given(
    label=st.sampled_from(_BOOLEAN_HEURISTIC_LABELS),
)
def test_witness_opinion_boolean_well_formed(label: str) -> None:
    policy = OthelloGradedPolicy()
    probe = _probe("d3")
    opinion = policy.witness_opinion(probe=probe, label=label, magnitude=None)
    _assert_opinion_well_formed(opinion)


@given(
    prefix=st.sampled_from(_COUNT_PREFIXES),
    magnitude=st.integers(min_value=1, max_value=64),
)
def test_witness_opinion_count_well_formed(prefix: str, magnitude: int) -> None:
    policy = OthelloGradedPolicy()
    probe = _probe("d3")
    label = f"{prefix}:{magnitude}"
    opinion = policy.witness_opinion(
        probe=probe, label=label, magnitude=magnitude
    )
    _assert_opinion_well_formed(opinion)


@given(
    prefix=st.sampled_from(_COUNT_PREFIXES),
    n=st.integers(min_value=1, max_value=63),
)
def test_count_opinion_uncertainty_decreases_with_n(prefix: str, n: int) -> None:
    """``u(n+1) <= u(n)`` for a COUNT label: more evidence -> less doubt.

    The beta-binomial conjugate prior :func:`Opinion.from_evidence`
    drives uncertainty monotone-down as the positive-evidence count
    grows: each additional observation tightens the belief band.
    """
    policy = OthelloGradedPolicy()
    probe = _probe("d3")
    label_n = f"{prefix}:{n}"
    label_n1 = f"{prefix}:{n + 1}"
    op_n = policy.witness_opinion(probe=probe, label=label_n, magnitude=n)
    op_n1 = policy.witness_opinion(
        probe=probe, label=label_n1, magnitude=n + 1
    )
    assert op_n1.u <= op_n.u


def test_with_probes_returns_new_policy() -> None:
    """``with_probes`` is Protocol-immutable: returns a fresh instance."""
    policy = OthelloGradedPolicy()
    probes = [_probe("d3", child_eval=1), _probe("c4", child_eval=-2)]
    bound = policy.with_probes(probes)
    assert bound is not policy


@given(
    child_evals=st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=1,
        max_size=10,
    ),
)
def test_with_probes_idempotent(child_evals: list[int]) -> None:
    """Binding twice with the same probes produces equivalent rank-fractions.

    The :meth:`with_probes` cache is a pure function of the probe tuple
    -- two binds against the same probes produce equal ``child_eval``
    rank-fraction dicts. ``move_base_rate`` is therefore stable across
    re-binds.
    """
    probes = [_probe(f"m{i}", child_eval=ev) for i, ev in enumerate(child_evals)]
    once = OthelloGradedPolicy().with_probes(probes)
    twice = once.with_probes(probes)
    for probe in probes:
        assert once.move_base_rate(probe) == twice.move_base_rate(probe)


def test_edge_trust_unchanged() -> None:
    """``edge_trust`` is ``Opinion.dogmatic_true(MAX_ENT_PRIOR)``."""
    policy = OthelloGradedPolicy()
    expected = Opinion.dogmatic_true(MAX_ENT_PRIOR)
    actual = policy.edge_trust
    assert actual.b == expected.b
    assert actual.d == expected.d
    assert actual.u == expected.u
    assert actual.a == expected.a


def test_with_probes_empty_returns_neutral_base_rate() -> None:
    """An unbound policy returns the neutral max-entropy prior."""
    policy = OthelloGradedPolicy()
    probe = _probe("d3", child_eval=5)
    assert policy.move_base_rate(probe) == MAX_ENT_PRIOR


def test_move_base_rate_ranks_ascending() -> None:
    """Larger ``child_eval`` (better for mover) -> larger rank-fraction.

    Othello orientation is mover-relative ASCENDING (same as chess):
    the probe with the largest ``child_eval`` is the best move and
    receives the highest rank-fraction.
    """
    probes = [
        _probe("worst", child_eval=-10),
        _probe("mid", child_eval=0),
        _probe("best", child_eval=10),
    ]
    bound = OthelloGradedPolicy().with_probes(probes)
    worst, mid, best = (
        bound.move_base_rate(p) for p in probes
    )
    assert worst < mid < best


def test_with_probes_mid_rank_ties() -> None:
    """Equal-``child_eval`` probes share the same rank-fraction (mid-rank)."""
    probes = [
        _probe("a", child_eval=1),
        _probe("b", child_eval=1),
        _probe("c", child_eval=5),
    ]
    bound = OthelloGradedPolicy().with_probes(probes)
    assert bound.move_base_rate(probes[0]) == bound.move_base_rate(probes[1])
    # The unique top-eval probe gets a strictly larger rank-fraction.
    assert bound.move_base_rate(probes[2]) > bound.move_base_rate(probes[0])


def test_witness_opinion_unknown_label_raises() -> None:
    """An unrecognised label is a typing failure -- raise, do not vacuous."""
    policy = OthelloGradedPolicy()
    probe = _probe("d3")
    with pytest.raises(ValueError):
        policy.witness_opinion(
            probe=probe, label="pro:unknown:concept", magnitude=None
        )
    with pytest.raises(ValueError):
        policy.witness_opinion(
            probe=probe, label="obj:unknown:metric:5", magnitude=5
        )


# --- Constants audit -------------------------------------------------------


def test_max_ent_prior_is_only_literal() -> None:
    """The ``graded_policy.py`` source contains only ``0.5`` / ``1`` / ``0``.

    Inspected by AST walk so a tuned constant cannot slip in disguised
    behind formatting. The acceptable set is: ``0.5`` (MAX_ENT_PRIOR
    and the Opinion.dogmatic_true argument); ``0.0`` / ``1.0`` (the
    beta-binomial arguments to ``Opinion.from_evidence`` and the
    Hazen ``2.0`` denominator); ``2.0`` (the Hazen mid-rank divisor);
    and small integer literals used as Hazen arithmetic indices.
    """
    source = Path(
        __import__("dialectical_othello.graded_policy", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    float_literals: set[float] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            float_literals.add(node.value)
    # The only float literals allowed are the MAX_ENT_PRIOR value
    # (0.5) and the beta-binomial / Hazen arithmetic primitives.
    allowed_floats: frozenset[float] = frozenset({0.0, 0.5, 1.0, 2.0})
    extra = float_literals - allowed_floats
    assert not extra, (
        f"unexpected float literal(s) in graded_policy.py: {extra}; "
        f"only MAX_ENT_PRIOR (0.5) and Hazen arithmetic primitives "
        f"are allowed"
    )
    # MAX_ENT_PRIOR itself MUST be the only non-Hazen float.
    assert MAX_ENT_PRIOR == 0.5
