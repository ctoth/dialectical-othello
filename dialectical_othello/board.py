"""``OthelloBoard``, ``OthelloMove`` — bitboard substrate for the Othello cartridge.

The cartridge's Board substrate satisfies the ``dialectical_games`` ``Board`` /
``Move`` Protocols (board.py:36-63 / 19-33). The plan (§1, §2, §8) pins:

- Frozen-dataclass bitboard — two ``uint64``s (one per colour) + ``turn: str``
  ("B" or "W") + ``pass_count: int`` (0, 1, or 2; game terminal at 2).
- Roll-our-own move-gen: 8-direction-walk, no PyPI dependency.
- ``to_fen()`` per the plan §2 convention (no standard Othello FEN exists).
- ``OthelloMove.move_id()`` returns algebraic notation (``"d3"``, ``"pass"``).
- ``apply()`` returns a new board with discs flipped, turn switched, and an
  auto-pass on the next ply if the next player has no legal moves.

Bit layout — PINNED at first commit (plan §R6):

    bit = file + 8 * rank        # file 0..7 = a..h, rank 0..7 = 1..8

So bit 0 is a1, bit 7 is h1, bit 56 is a8, bit 63 is h8. This is the
"LSB = a1, ranks-up" convention. ``to_fen()`` prints rank 8 first (top row of
a printed board), rank 1 last; within each row file a is leftmost. The
serialised form is the external witness to this choice — anyone reading
``to_fen()`` output infers the layout from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# --- Geometry ---------------------------------------------------------------

NUM_SQUARES = 64
BOARD_MASK = (1 << 64) - 1

# Files / ranks. ``FILE_A`` is the column on file a (bits 0, 8, 16, ...);
# ``FILE_H`` the column on file h.
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
NOT_FILE_A = BOARD_MASK ^ FILE_A
NOT_FILE_H = BOARD_MASK ^ FILE_H

# The 8 direction-shift functions. A direction shifts a bitboard one square
# in that direction; squares that would wrap off an edge are masked away.
# N is +rank (+8); S is -rank (-8); E is +file (+1); W is -file (-1).
def _shift_n(b: int) -> int:
    return (b << 8) & BOARD_MASK


def _shift_s(b: int) -> int:
    return b >> 8


def _shift_e(b: int) -> int:
    return (b << 1) & NOT_FILE_A & BOARD_MASK


def _shift_w(b: int) -> int:
    return (b >> 1) & NOT_FILE_H


def _shift_ne(b: int) -> int:
    return (b << 9) & NOT_FILE_A & BOARD_MASK


def _shift_nw(b: int) -> int:
    return (b << 7) & NOT_FILE_H & BOARD_MASK


def _shift_se(b: int) -> int:
    return (b >> 7) & NOT_FILE_A


def _shift_sw(b: int) -> int:
    return (b >> 9) & NOT_FILE_H


#: The eight direction-shift functions, in (N, NE, E, SE, S, SW, W, NW) order.
_SHIFTS: tuple[Callable[[int], int], ...] = (
    _shift_n,
    _shift_ne,
    _shift_e,
    _shift_se,
    _shift_s,
    _shift_sw,
    _shift_w,
    _shift_nw,
)


def _algebraic(square: int) -> str:
    """Return the algebraic notation (e.g. ``"d3"``) for a 0..63 square index."""
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square {square} out of range 0..63")
    file_ch = "abcdefgh"[square & 7]
    rank_ch = str((square >> 3) + 1)
    return f"{file_ch}{rank_ch}"


def _square_from_algebraic(text: str) -> int:
    """Parse an algebraic square (``"d3"``) back to the 0..63 index."""
    if len(text) != 2:
        raise ValueError(f"algebraic square {text!r} must be 2 chars")
    file_ch, rank_ch = text[0], text[1]
    if file_ch not in "abcdefgh":
        raise ValueError(f"bad file {file_ch!r} in {text!r}")
    if rank_ch not in "12345678":
        raise ValueError(f"bad rank {rank_ch!r} in {text!r}")
    file_idx = "abcdefgh".index(file_ch)
    rank_idx = int(rank_ch) - 1
    return file_idx + 8 * rank_idx


# --- Move type --------------------------------------------------------------


@dataclass(frozen=True, order=True)
class OthelloMove:
    """An Othello move (plan §2).

    ``square`` is the 0..63 bit index of the placed disc, or ``None`` for a
    pass move (no legal disc-flip available). Algebraic move_id is the
    universal Othello notation (e.g. ``"d3"``, ``"pass"``).
    """

    square: int | None

    @classmethod
    def pass_move(cls) -> OthelloMove:
        """Return the singleton pass move."""
        return cls(square=None)

    def is_pass(self) -> bool:
        """True iff this is a pass move."""
        return self.square is None

    def move_id(self) -> str:
        """The Move-Protocol identifier — algebraic notation or ``"pass"``."""
        if self.square is None:
            return "pass"
        return _algebraic(self.square)


# --- Initial position -------------------------------------------------------


def _initial_black() -> int:
    """Standard Othello start: Black discs on e4 (bit 28) and d5 (bit 35)."""
    return (1 << _square_from_algebraic("e4")) | (1 << _square_from_algebraic("d5"))


def _initial_white() -> int:
    """Standard Othello start: White discs on d4 (bit 27) and e5 (bit 36)."""
    return (1 << _square_from_algebraic("d4")) | (1 << _square_from_algebraic("e5"))


_OPPONENT = {"B": "W", "W": "B"}


# --- Board ------------------------------------------------------------------


@dataclass(frozen=True)
class OthelloBoard:
    """Immutable bitboard Othello position (plan §1, §2).

    ``black`` and ``white`` are uint64 bitboards; bit ``file + 8*rank`` is set
    iff that square holds a disc of the corresponding colour. ``turn`` is
    ``"B"`` or ``"W"``. ``pass_count`` is 0, 1, or 2; the game is terminal
    when it reaches 2 (both sides passed in succession).
    """

    black: int = 0
    white: int = 0
    turn: str = "B"
    pass_count: int = 0

    # -- construction --------------------------------------------------------

    @classmethod
    def initial(cls) -> OthelloBoard:
        """Return the standard Othello starting position, Black to move."""
        return cls(black=_initial_black(), white=_initial_white(), turn="B", pass_count=0)

    # -- bitboard helpers ----------------------------------------------------

    def _mover_opponent(self) -> tuple[int, int]:
        """Return ``(mover_bb, opponent_bb)`` for the side to move."""
        if self.turn == "B":
            return self.black, self.white
        return self.white, self.black

    def _empty(self) -> int:
        """The bitboard of empty squares."""
        return BOARD_MASK ^ (self.black | self.white)

    def _legal_moves_bb(self) -> int:
        """Compute the bitboard of legal disc-placement squares for the mover.

        Standard direction-walk: for each of the eight directions, slide the
        mover's bitboard against contiguous opponent discs and AND the final
        slide with the empty squares. Any bit lit after AND is a square where
        placing a disc would bracket at least one opponent disc in that
        direction.
        """
        mover, opp = self._mover_opponent()
        empty = self._empty()
        legal = 0
        for shift in _SHIFTS:
            # First step: a mover-adjacent opponent disc in this direction.
            candidates = shift(mover) & opp
            # Walk further opponent discs (up to 5 more — board is 8 wide).
            for _ in range(5):
                candidates |= shift(candidates) & opp
            # The square one further past the run, if empty, is a legal move.
            legal |= shift(candidates) & empty
        return legal

    # -- public surface ------------------------------------------------------

    @property
    def empty_count(self) -> int:
        """Count of empty squares on the board."""
        return 64 - (self.black | self.white).bit_count()

    def disc_counts(self) -> tuple[int, int]:
        """Return ``(black_count, white_count)``."""
        return self.black.bit_count(), self.white.bit_count()

    def is_terminal(self) -> bool:
        """True iff the game has ended (two consecutive passes)."""
        return self.pass_count >= 2

    def legal_moves(self) -> tuple[OthelloMove, ...]:
        """Return the legal moves for the side to move.

        Returns disc placements only — passes are handled automatically by
        :meth:`apply`, so a caller never sees a pass move in this tuple. An
        empty tuple means the position is terminal (both sides have no
        legal disc-flip, i.e. ``pass_count >= 2``).
        """
        if self.is_terminal():
            return ()
        legal_bb = self._legal_moves_bb()
        if legal_bb == 0:
            # The mover has no disc-flips. In a non-terminal board this can
            # only occur right after construction from a serialised position
            # — :meth:`apply` always advances past forced passes so a
            # ``legal_moves()`` caller on an :meth:`apply`-produced board
            # never reaches this branch. Surface the forced pass to the
            # caller so they can drive the game forward.
            return (OthelloMove.pass_move(),)
        moves: list[OthelloMove] = []
        bb = legal_bb
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            moves.append(OthelloMove(square=sq))
            bb ^= lsb
        return tuple(moves)

    def _flip_bb(self, square: int) -> int:
        """Return the bitboard of opponent discs that would flip if the mover
        played a disc at ``square``. ``square`` must be empty and a legal
        placement; for an illegal square the result may be 0."""
        mover, opp = self._mover_opponent()
        placed = 1 << square
        flips = 0
        for shift in _SHIFTS:
            # Collect contiguous opponent discs in this direction starting
            # from ``placed``.
            run = 0
            cur = shift(placed) & opp
            while cur:
                run |= cur
                cur = shift(cur) & opp
            # The square past the run must be a mover disc to bracket.
            end = shift(run) if run else 0
            if end & mover:
                flips |= run
        return flips

    def apply(self, move: OthelloMove) -> OthelloBoard:
        """Return a new board with ``move`` played and turn switched.

        Three cases:

        - **Placement move** (``move.square is not None``): places the disc,
          flips bracketed opponent runs, switches turn, resets ``pass_count``
          to 0. THEN if the next side has no disc-flip *and* the side that
          just moved still has a disc-flip on the resulting position, the
          opponent's pass is auto-applied — turn switches back, ``pass_count``
          becomes 1. If NEITHER side has a disc-flip on the resulting
          position, ``pass_count`` becomes 2 (terminal).
        - **Pass move** (``move.square is None``): legal only when the mover
          has no disc-flip; switches turn, increments ``pass_count``.

        The auto-pass rule means a caller never has to play a pass
        explicitly after a placement: the placement encodes whatever forced
        passes follow.
        """
        if self.is_terminal():
            raise ValueError("cannot apply a move to a terminal board")
        if move.is_pass():
            # Pass legality: the mover must genuinely have no disc-flip.
            if self._legal_moves_bb() != 0:
                raise ValueError(
                    "pass move is illegal — mover has at least one disc-flip"
                )
            return OthelloBoard(
                black=self.black,
                white=self.white,
                turn=_OPPONENT[self.turn],
                pass_count=self.pass_count + 1,
            )
        square = move.square
        assert square is not None  # narrowed by is_pass() above
        if not 0 <= square < NUM_SQUARES:
            raise ValueError(f"move square {square} out of range")
        placed = 1 << square
        if placed & (self.black | self.white):
            raise ValueError(f"square {square} is already occupied")
        flips = self._flip_bb(square)
        if flips == 0:
            raise ValueError(
                f"move {move.move_id()} is illegal — no opponent disc would flip"
            )
        if self.turn == "B":
            new_black = self.black | placed | flips
            new_white = self.white & ~flips
        else:
            new_white = self.white | placed | flips
            new_black = self.black & ~flips
        next_turn = _OPPONENT[self.turn]
        next_board = OthelloBoard(
            black=new_black,
            white=new_white,
            turn=next_turn,
            pass_count=0,
        )
        if next_board._legal_moves_bb() != 0:
            return next_board
        # Opponent has no disc-flip on the next board. Auto-pass — flip turn
        # back to the side that just moved, increment ``pass_count``. If the
        # side that just moved also has no disc-flip on the new board (i.e.
        # the board is full or both sides are blocked), ``pass_count``
        # becomes 2 and the game is terminal.
        passed_back = OthelloBoard(
            black=new_black,
            white=new_white,
            turn=self.turn,
            pass_count=1,
        )
        if passed_back._legal_moves_bb() != 0:
            return passed_back
        return OthelloBoard(
            black=new_black,
            white=new_white,
            turn=next_turn,
            pass_count=2,
        )

    # -- serialisation -------------------------------------------------------

    def to_fen(self) -> str:
        """Serialise to the plan §2 form (no standard Othello FEN exists).

        Format (verbatim):

            <row 8>/<row 7>/<row 6>/<row 5>/<row 4>/<row 3>/<row 2>/<row 1> <turn> <pass_count>

        Each row is 8 chars; each char is ``'.'`` (empty), ``'B'`` (black
        disc), or ``'W'`` (white disc). Rows are printed top-down (rank 8 first,
        rank 1 last). Within a row file a is leftmost, file h rightmost.
        ``<turn>`` is ``'B'`` or ``'W'``; ``<pass_count>`` is ``'0'``, ``'1'``,
        or ``'2'``. Example (standard Othello starting position, Black to
        move, no passes — d5 is Black and e5 White on rank 5, d4 White and
        e4 Black on rank 4):

            ``"......../......../......../...BW.../...WB.../......../......../........ B 0"``
        """
        rows: list[str] = []
        for rank in range(7, -1, -1):  # rank 8 (index 7) first
            chars: list[str] = []
            for file in range(8):
                bit = 1 << (file + 8 * rank)
                if self.black & bit:
                    chars.append("B")
                elif self.white & bit:
                    chars.append("W")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        return f"{'/'.join(rows)} {self.turn} {self.pass_count}"

    @classmethod
    def from_fen(cls, fen: str) -> OthelloBoard:
        """Parse a board serialised by :meth:`to_fen`.

        Accepts the exact form documented on :meth:`to_fen` — the three
        whitespace-separated fields ``<rows> <turn> <pass_count>``, with rows
        ``/``-separated top-down.
        """
        parts = fen.strip().split()
        if len(parts) != 3:
            raise ValueError(f"malformed Othello FEN, expected 3 fields: {fen!r}")
        rows_text, turn, pass_text = parts
        if turn not in ("B", "W"):
            raise ValueError(f"bad turn {turn!r} in Othello FEN")
        try:
            pass_count = int(pass_text)
        except ValueError:
            raise ValueError(f"bad pass_count {pass_text!r} in Othello FEN") from None
        if pass_count not in (0, 1, 2):
            raise ValueError(f"pass_count {pass_count} not in 0..2 in Othello FEN")
        rows = rows_text.split("/")
        if len(rows) != 8:
            raise ValueError(
                f"Othello FEN must have 8 rank rows, got {len(rows)}: {fen!r}"
            )
        black = 0
        white = 0
        for row_idx, row in enumerate(rows):
            if len(row) != 8:
                raise ValueError(
                    f"rank row {row!r} has length {len(row)} != 8 in {fen!r}"
                )
            rank = 7 - row_idx  # rows[0] is rank 8 (rank index 7)
            for file, ch in enumerate(row):
                bit = 1 << (file + 8 * rank)
                if ch == "B":
                    black |= bit
                elif ch == "W":
                    white |= bit
                elif ch == ".":
                    pass
                else:
                    raise ValueError(f"bad cell {ch!r} in {fen!r}")
        return cls(black=black, white=white, turn=turn, pass_count=pass_count)
