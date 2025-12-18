#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import Piece
from constants import TOTAL_COUNTS, PIECE_JP

def compute_gote_remaining(
    board_map: Dict[Tuple[int, int], Optional[Piece]],
    sente_hand: Dict[str, int],
) -> Dict[str, int]:
    used = {k: 0 for k in TOTAL_COUNTS.keys()}
    for p in board_map.values():
        if p is None:
            continue
        if p.kind in used:
            used[p.kind] += 1

    for kind, n in sente_hand.items():
        if kind in used:
            used[kind] += n

    remaining: Dict[str, int] = {}
    for kind, total in TOTAL_COUNTS.items():
        if kind == "K":
            continue
        left = total - used.get(kind, 0)
        if left > 0:
            remaining[kind] = left
    return remaining


def hands_to_sfen(hands_b: Dict[str, int], hands_w: Dict[str, int]) -> str:
    order = ["R", "B", "G", "S", "N", "L", "P"]
    parts: List[str] = []

    def add(kind: str, n: int, is_black: bool) -> None:
        if n <= 0:
            return
        c = kind if is_black else kind.lower()
        parts.append(c if n == 1 else f"{n}{c}")

    for k in order:
        add(k, hands_b.get(k, 0), True)
    for k in order:
        add(k, hands_w.get(k, 0), False)

    return "-" if not parts else "".join(parts)


def board_to_sfen(board_map: Dict[Tuple[int, int], Optional[Piece]]) -> str:
    rows: List[str] = []
    for r in range(1, 10):
        empties = 0
        row = ""
        for f in range(9, 0, -1):
            p = board_map[(f, r)]
            if p is None:
                empties += 1
                continue
            if empties:
                row += str(empties)
                empties = 0
            if p.prom:
                row += "+"
            ch = p.kind
            row += ch if p.color == "B" else ch.lower()
        if empties:
            row += str(empties)
        rows.append(row)
    return "/".join(rows)


def snapshot_to_sfen(
    board_map: Dict[Tuple[int, int], Optional[Piece]],
    hands_b: Dict[str, int],
    side_to_move: str,
    gote_hands_auto: Dict[str, int],
) -> str:
    board_part = board_to_sfen(board_map)
    turn = "b" if side_to_move == "B" else "w"
    hands_part = hands_to_sfen(hands_b, gote_hands_auto)
    return f"{board_part} {turn} {hands_part} 1"

def sfen_to_snapshot(sfen: str):
    """Parse SFEN into (board_map, hands_b, side_to_move). Ignores gote hands (we auto-compute)."""
    parts = sfen.strip().split()
    if len(parts) < 3:
        raise ValueError("SFEN形式が不正です")
    board_part, turn_part, hands_part = parts[0], parts[1], parts[2]

    # board
    board_map: Dict[Tuple[int,int], Optional[Piece]] = {(f,r):None for f in range(1,10) for r in range(1,10)}
    rows = board_part.split("/")
    if len(rows) != 9:
        raise ValueError("SFEN盤面の段数が不正です")
    for r_idx, row in enumerate(rows, start=1):  # r=1..9
        f = 9
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                n = int(ch)
                f -= n
                i += 1
                continue
            prom = False
            if ch == "+":
                prom = True
                i += 1
                ch = row[i]
            color = "B" if ch.isupper() else "W"
            kind = ch.upper()
            if kind not in PIECE_JP:
                raise ValueError("SFEN駒種が不正です")
            board_map[(f, r_idx)] = Piece(color=color, kind=kind, prom=prom)
            f -= 1
            i += 1
        if f != 0:
            # f should end at 0 after filling 9 files
            pass

    side_to_move = "B" if turn_part == "b" else "W"

    # hands (black only)
    hands_b: Dict[str,int] = {}
    if hands_part != "-":
        i = 0
        while i < len(hands_part):
            # count may be multiple digits
            if hands_part[i].isdigit():
                j = i
                while j < len(hands_part) and hands_part[j].isdigit():
                    j += 1
                cnt = int(hands_part[i:j])
                pch = hands_part[j]
                i = j + 1
            else:
                cnt = 1
                pch = hands_part[i]
                i += 1
            if pch.isupper():
                k = pch
                if k in PIECE_JP and k != "K":
                    hands_b[k] = hands_b.get(k, 0) + cnt
            # lowercase (gote) is ignored because we auto-compute at output
    return board_map, hands_b, side_to_move
