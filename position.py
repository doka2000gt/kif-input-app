#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal
import copy
import re

from models import Piece, Move
from constants import (
    PIECE_JP,
    PROMOTABLE,
    KIND_TO_PYO,
    RANK_KANJI,
)
from helpers import inv_count_kanji


__all__ = ["ShogiPosition"]


Square = Tuple[int, int]
Color = Literal["B", "W"]

# parse_numeric の戻り値（現状の仕様に合わせて厳密化）
NumericMode = Literal["drop_pick", "move"]
ParseNumericResult = Tuple[
    NumericMode,
    None,                 # 旧仕様の「予約枠」。今は常に None を返しているので明示
    Optional[Square],     # from_sq（drop のとき None）
    Square,               # to_sq
    bool,                 # promote
]

# clone_state の戻り値
Snapshot = Tuple[
    Dict[Square, Optional[Piece]],          # board
    Dict[Color, Dict[str, int]],            # hands
    Color,                                   # side_to_move
    List[Move],                              # moves
]


class ShogiPosition:
    def __init__(self) -> None:
        self.board: Dict[Square, Optional[Piece]] = {(f, r): None for f in range(1, 10) for r in range(1, 10)}
        self.hands: Dict[Color, Dict[str, int]] = {"B": {}, "W": {}}
        self.side_to_move: Color = "B"
        self.moves: List[Move] = []
        self._history: List[Snapshot] = []

    def clone_state(self) -> Snapshot:
        return (
            copy.deepcopy(self.board),
            copy.deepcopy(self.hands),
            self.side_to_move,
            copy.deepcopy(self.moves),
        )

    def push_history(self) -> None:
        self._history.append(self.clone_state())

    def undo(self) -> bool:
        if not self._history:
            return False
        self.board, self.hands, self.side_to_move, self.moves = self._history.pop()
        return True

    def clear_moves(self) -> None:
        self.moves = []
        self._history = []

    def clear_all(self) -> None:
        self.board = {(f, r): None for f in range(1, 10) for r in range(1, 10)}
        self.hands = {"B": {}, "W": {}}
        self.side_to_move = "B"
        self.clear_moves()

    # ---- setup editing ----
    def set_piece(self, sq: Square, p: Optional[Piece]) -> None:
        self.board[sq] = p

    def set_hand(self, color: Color, kind: str, n: int) -> None:
        if n <= 0:
            self.hands[color].pop(kind, None)
        else:
            self.hands[color][kind] = n

    # ---- numeric input ----
    def parse_numeric(self, s: str) -> ParseNumericResult:
        s = s.strip()
        if not re.fullmatch(r"\d{3,5}", s):
            raise ValueError("数字入力は 4桁(移動) / 5桁(成り) / 3桁(打ち:0+先2桁) です")

        # drop: 0 + to(2 digits) e.g. 076
        if len(s) == 3 and s[0] == "0":
            to_file = int(s[1])
            to_rank = int(s[2])
            if not (1 <= to_file <= 9 and 1 <= to_rank <= 9):
                raise ValueError("マスは 11〜99 の範囲です")
            return ("drop_pick", None, None, (to_file, to_rank), False)

        # move: 4 or 5 digits, last digit 1 means promote
        promote = (len(s) == 5 and s[-1] == "1")
        core = s[:4]
        f1, r1, f2, r2 = map(int, core)
        for x in (f1, r1, f2, r2):
            if not (1 <= x <= 9):
                raise ValueError("マスは 11〜99 の範囲です")
        return ("move", None, (f1, r1), (f2, r2), promote)

    # ---- minimal legality (for editing convenience) ----
    # We intentionally keep it permissive; strict check is via python-shogi when available.

    def _remove_from_hand(self, color: Color, kind: str) -> None:
        c = self.hands[color].get(kind, 0)
        if c <= 0:
            raise ValueError("持ち駒がありません")
        if c == 1:
            del self.hands[color][kind]
        else:
            self.hands[color][kind] = c - 1

    def _add_to_hand(self, color: Color, kind: str) -> None:
        self.hands[color][kind] = self.hands[color].get(kind, 0) + 1

    def drop_candidates(self, to: Square) -> List[str]:
        if self.board[to] is not None:
            return []
        # Prefer R,B,G,S,N,L,P order
        order = "RBGSLNP"
        kinds = sorted(self.hands[self.side_to_move].keys(), key=lambda k: order.find(k) if k in order else 99)
        result: List[str] = []
        for k in kinds:
            # very light nifu check
            if k == "P":
                file_, _ = to
                nifu = False
                for r in range(1, 10):
                    p = self.board[(file_, r)]
                    if p and p.color == self.side_to_move and p.kind == "P" and not p.prom:
                        nifu = True
                        break
                if nifu:
                    continue
            result.append(k)
        return result

    def apply_move_minimal(
        self,
        kind: str,
        frm: Optional[Square],
        to: Square,
        promote: bool,
        is_drop: bool,
    ) -> Move:
        self.push_history()
        prev_to = self.moves[-1].to_sq if self.moves else None
        same = (prev_to == to)

        if is_drop:
            self._remove_from_hand(self.side_to_move, kind)
            if self.board[to] is not None:
                self.undo()
                raise ValueError("打ち先に駒があります")
            self.board[to] = Piece(self.side_to_move, kind, False)
            mv = Move(True, kind, None, to, False, same)
        else:
            p = self.board.get(frm)
            if p is None or p.color != self.side_to_move:
                self.undo()
                raise ValueError("移動元に手番の駒がありません")
            dest = self.board[to]
            if dest is not None:
                if dest.color == self.side_to_move:
                    self.undo()
                    raise ValueError("移動先に自分の駒があります")
                self._add_to_hand(self.side_to_move, dest.kind)

            self.board[frm] = None
            np = copy.deepcopy(p)
            if promote:
                if np.kind not in PROMOTABLE or np.prom:
                    self.undo()
                    raise ValueError("成れません")
                np.prom = True
            self.board[to] = np
            mv = Move(False, np.kind, frm, to, promote, same)

        self.side_to_move = "W" if self.side_to_move == "B" else "B"
        self.moves.append(mv)
        return mv

    # ---- output helpers ----
    def hands_to_piyo(self, color: Color) -> str:
        order = ["R", "B", "G", "S", "N", "L", "P"]
        parts: List[str] = []
        for k in order:
            n = self.hands[color].get(k, 0)
            if n <= 0:
                continue
            parts.append(PIECE_JP[k] + inv_count_kanji(n))
        return " ".join(parts) + (" " if parts else "")

    def board_to_piyo(self) -> str:
        lines: List[str] = []
        lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
        lines.append("+---------------------------+")
        for r in range(1, 10):
            row: List[str] = []
            for f in range(9, 0, -1):
                p = self.board[(f, r)]
                if p is None:
                    row.append(" ・")
                else:
                    name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
                    cell = ("v" + name) if p.color == "W" else (" " + name)
                    row.append(cell)
            lines.append("|" + "".join(row) + f"|{RANK_KANJI[r]}")
        lines.append("+---------------------------+")
        return "\n".join(lines)
