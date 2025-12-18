#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Tuple
import copy

from models import Piece, Move
from constants import PIECE_JP, KIND_TO_PYO, RANK_KANJI
from helpers import sq_to_kif, sq_to_paren


Square = Tuple[int, int]

__all__ = [
    "board_map_to_piyo",
    "kif_line_for_minimal_move",
    "apply_minimal_to_tmp",
]


def board_map_to_piyo(board_map: Dict[Square, Optional[Piece]]) -> str:
    lines = []
    lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
    lines.append("+---------------------------+")
    for r in range(1, 10):
        row = []
        for f in range(9, 0, -1):
            p = board_map[(f, r)]
            if p is None:
                row.append(" ・")
            else:
                name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
                cell = ("v" + name) if p.color == "W" else (" " + name)
                row.append(cell)
        lines.append("|" + "".join(row) + f"|{RANK_KANJI[r]}")
    lines.append("+---------------------------+")
    return "\n".join(lines)


def kif_line_for_minimal_move(
    idx: int,
    mv: Move,
    prev_to: Optional[Square],
    sec: int,
    total_sec: int,
) -> Tuple[str, Square]:
    dst = "同" if (prev_to is not None and prev_to == mv.to_sq) else sq_to_kif(*mv.to_sq)
    if mv.is_drop:
        body = f"{dst}{PIECE_JP[mv.kind]}打"
    else:
        name = PIECE_JP[mv.kind] + ("成" if mv.promote else "")
        body = f"{dst}{name}{sq_to_paren(*mv.from_sq)}"
    time_part = f"( 0:{sec:02d}/00:00:{total_sec:02d})"
    line = f"{idx:4d} {body:<12} {time_part}"
    return line, mv.to_sq


def apply_minimal_to_tmp(
    board_map: Dict[Square, Optional[Piece]],
    hands: Dict[str, Dict[str, int]],
    side: str,
    mv: Move,
) -> None:
    if mv.is_drop:
        # remove from hand if present
        if mv.kind in hands[side]:
            hands[side][mv.kind] -= 1
            if hands[side][mv.kind] <= 0:
                del hands[side][mv.kind]
        board_map[mv.to_sq] = Piece(side, mv.kind, False)
        return

    p = board_map[mv.from_sq]
    # capture -> add to hand
    dest = board_map[mv.to_sq]
    if dest is not None:
        hands[side][dest.kind] = hands[side].get(dest.kind, 0) + 1

    board_map[mv.from_sq] = None
    np = copy.deepcopy(p)
    if mv.promote:
        np.prom = True
    board_map[mv.to_sq] = np
