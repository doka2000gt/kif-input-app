#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import copy
import hashlib
import pathlib

from models import Piece, Move, SolveNode
from constants import RANK_KANJI, PIECE_JP, PROMOTED_JP, KIND_TO_PYO
from helpers import (
    sq_to_kif,
    sq_to_paren,
    inv_count_kanji,
    now_yyyy_mm_dd_hhmmss,
    format_total_time,
)
from paths import _resolve_kif_path

# python-shogi は optional にしておく（solver側と同じ方針）
try:
    import shogi  # pip install python-shogi
except Exception:
    shogi = None


def _sq_from_shogi_square(sq: int) -> Tuple[int, int]:
    file_ = 9 - (sq % 9)
    rank_ = (sq // 9) + 1
    return file_, rank_


def _piece_type_to_kind(pt: int) -> str:
    if shogi is None:
        return "P"
    mapping = {
        getattr(shogi, "PAWN", None): "P",
        getattr(shogi, "LANCE", None): "L",
        getattr(shogi, "KNIGHT", None): "N",
        getattr(shogi, "SILVER", None): "S",
        getattr(shogi, "GOLD", None): "G",
        getattr(shogi, "BISHOP", None): "B",
        getattr(shogi, "ROOK", None): "R",
        getattr(shogi, "KING", None): "K",
        getattr(shogi, "PRO_PAWN", None): "P",
        getattr(shogi, "PRO_LANCE", None): "L",
        getattr(shogi, "PRO_KNIGHT", None): "N",
        getattr(shogi, "PRO_SILVER", None): "S",
        getattr(shogi, "HORSE", None): "B",
        getattr(shogi, "DRAGON", None): "R",
    }
    mapping = {k: v for k, v in mapping.items() if k is not None}
    return mapping.get(pt, "P")


def _is_promoted_piece_type(pt: int) -> bool:
    if shogi is None:
        return False
    promoted = {
        getattr(shogi, "PRO_PAWN", None),
        getattr(shogi, "PRO_LANCE", None),
        getattr(shogi, "PRO_KNIGHT", None),
        getattr(shogi, "PRO_SILVER", None),
        getattr(shogi, "HORSE", None),
        getattr(shogi, "DRAGON", None),
    }
    return pt in {x for x in promoted if x is not None}


def kif_line_for_shogi_move(
    idx: int, board_before, move, prev_to: Optional[Tuple[int, int]], sec: int, total_sec: int
) -> Tuple[str, Tuple[int, int]]:
    to_file, to_rank = _sq_from_shogi_square(move.to_square)
    dst = "同" if (prev_to is not None and prev_to == (to_file, to_rank)) else sq_to_kif(to_file, to_rank)

    if move.drop_piece_type is not None:
        kind = _piece_type_to_kind(move.drop_piece_type)
        body = f"{dst}{PIECE_JP[kind]}打"
    else:
        p = board_before.piece_at(move.from_square)
        kind = _piece_type_to_kind(p.piece_type)
        if move.promotion:
            name = PIECE_JP[kind] + "成"
        else:
            if _is_promoted_piece_type(p.piece_type) and kind in PROMOTED_JP:
                name = PROMOTED_JP[kind]
            else:
                name = PIECE_JP[kind]
        fr_file, fr_rank = _sq_from_shogi_square(move.from_square)
        body = f"{dst}{name}{sq_to_paren(fr_file, fr_rank)}"

    time_part = f"( 0:{sec:02d}/00:00:{total_sec:02d})"
    line = f"{idx:4d} {body:<12} {time_part}"
    return line, (to_file, to_rank)


def _hands_to_line(hands: Dict[str, int]) -> str:
    order = ["R", "B", "G", "S", "N", "L", "P"]
    parts = []
    for k in order:
        n = hands.get(k, 0)
        if n <= 0:
            continue
        parts.append(f"{PIECE_JP[k]}{inv_count_kanji(n)}")
    return " ".join(parts) + (" " if parts else "")


def board0_to_piyo(board0) -> str:
    if shogi is None:
        raise RuntimeError("python-shogi が必要です（board0_to_piyo）")
    lines = []
    lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
    lines.append("+---------------------------+")
    for r in range(1, 10):
        row = []
        for f in range(9, 0, -1):
            sq = (r - 1) * 9 + (9 - f)
            p = board0.piece_at(sq)
            if p is None:
                row.append(" ・")
            else:
                kind = _piece_type_to_kind(p.piece_type)
                name = PROMOTED_JP[kind] if (_is_promoted_piece_type(p.piece_type) and kind in PROMOTED_JP) else PIECE_JP[kind]
                cell = ("v" + name) if p.color == shogi.WHITE else (" " + name)
                row.append(cell)
        lines.append("|" + "".join(row) + f"|{RANK_KANJI[r]}")
    lines.append("+---------------------------+")
    return "\n".join(lines)


def _board_map_to_piyo(board_map: Dict[Tuple[int, int], Optional[Piece]]) -> str:
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


def _kif_line_for_minimal_move(idx: int, board_map, mv: Move, prev_to, sec: int, total_sec: int):
    dst = "同" if (prev_to is not None and prev_to == mv.to_sq) else sq_to_kif(*mv.to_sq)
    if mv.is_drop:
        body = f"{dst}{PIECE_JP[mv.kind]}打"
    else:
        name = PIECE_JP[mv.kind] + ("成" if mv.promote else "")
        body = f"{dst}{name}{sq_to_paren(*mv.from_sq)}"
    time_part = f"( 0:{sec:02d}/00:00:{total_sec:02d})"
    line = f"{idx:4d} {body:<12} {time_part}"
    return line, mv.to_sq


def _apply_minimal_to_tmp(board_map, hands, side, mv: Move):
    if mv.is_drop:
        if mv.kind in hands[side]:
            hands[side][mv.kind] -= 1
            if hands[side][mv.kind] <= 0:
                del hands[side][mv.kind]
        board_map[mv.to_sq] = Piece(side, mv.kind, False)
    else:
        p = board_map[mv.from_sq]
        dest = board_map[mv.to_sq]
        if dest is not None:
            hands[side][dest.kind] = hands[side].get(dest.kind, 0) + 1
        board_map[mv.from_sq] = None
        np = copy.deepcopy(p)
        if mv.promote:
            np.prom = True
        board_map[mv.to_sq] = np


def generate_kif_single_line(board0, hands_b: Dict[str,int], gote_hands_auto: Dict[str,int],
                             sente_name: str, gote_name: str, moves: List[object], outfile: str,
                             seen: Optional[Dict[str,str]] = None) -> Optional[pathlib.Path]:
    """Write a single-line KIF (no variations). If seen is given, skip identical outputs."""
    header: List[str] = []
    header.append("# ---- Kifu for Mac V0.53 夢の詰将棋メーカー by CUI ----")
    header.append(f"終了日時：{now_yyyy_mm_dd_hhmmss()}")
    header.append("手合割：平手")
    header.append("後手の持駒：" + _hands_to_line(gote_hands_auto))
    header.append(board0_to_piyo(board0))
    header.append("先手の持駒：" + _hands_to_line(hands_b))
    header.append(f"先手：{sente_name}")
    header.append(f"後手：{gote_name}")
    header.append("手数----指手---------消費時間--")

    board = copy.deepcopy(board0)
    prev_to = None
    sec_per_move = 3
    total_sec = 0
    lines: List[str] = []
    idx = 1
    for mv in moves:
        total_sec += sec_per_move
        line, prev_to = kif_line_for_shogi_move(idx, board, mv, prev_to, sec_per_move, total_sec)
        lines.append(line)
        board.push(mv)
        idx += 1

    # End line: 詰み
    end_line = f"{idx:4d} 詰み         ( 0:{sec_per_move:02d}/{format_total_time(total_sec)})"
    # Use the same cumulative time as last move; keep formatting simple.
    # Some viewers ignore time anyway; ANKIF tested OK with this style.
    lines.append(end_line)

    text = "\n".join(header + lines) + "\n"
    outp = _resolve_kif_path(outfile)

    # dedup: skip identical final content (by cp932 bytes)
    if seen is not None:
        key = hashlib.sha1(text.encode("cp932", errors="replace")).hexdigest()
        if key in seen:
            prev = pathlib.Path(seen[key]).name
            print(f"[dedup] {outp.name} は {prev} と同一なので省略")
            return None
        seen[key] = str(outp)
    outp.write_bytes(text.encode("cp932", errors="replace"))
    return outp


def generate_kif_with_variations(board0, hands_b: Dict[str,int], gote_hands_auto: Dict[str,int], side_to_move: str,
                                sente_name: str, gote_name: str, tree: SolveNode, outfile: str):
    # Header (match 015.kif style)
    header: List[str] = []
    header.append("# ---- Kifu for Mac V0.53 夢の詰将棋メーカー by CUI ----")
    header.append(f"終了日時：{now_yyyy_mm_dd_hhmmss()}")
    header.append("手合割：平手")
    # gote hands: auto
    header.append("後手の持駒：" + _hands_to_line(gote_hands_auto))
    header.append(board0_to_piyo(board0))
    header.append("先手の持駒：" + _hands_to_line(hands_b))
    header.append(f"先手：{sente_name}")
    header.append(f"後手：{gote_name}")
    header.append("手数----指手---------消費時間--")

    sec_per_move = 3

    # Mainline (first-child path from root)
    main_moves, variations = build_mainline_and_variations(board0, tree, start_idx=1)

    lines: List[str] = []
    board = copy.deepcopy(board0)
    prev_to = None
    total_sec = 0
    idx = 1
    for mv in main_moves:
        total_sec += sec_per_move
        line, prev_to = kif_line_for_shogi_move(idx, board, mv, prev_to, sec_per_move, total_sec)
        lines.append(line)
        board.push(mv)
        idx += 1

    # Append terminal "詰み" line (like 015.kif)
    lines.append(f"{idx:4d} 詰み         ( 0:{sec_per_move:02d}/00:00:{total_sec:02d})")

    # Emit variations (and nested variations) depth-first
    var_queue = variations[:]  # (div_idx, prefix_moves, subtree)
    # variations from root have prefix_moves = prefix along mainline (already in moves list), but we stored only relative prefix,
    # so rebuild full prefix from main_moves for those.
    fixed_queue: List[Tuple[int, List[object], SolveNode]] = []
    for div_idx, rel_prefix, subtree in var_queue:
        # rel_prefix is moves along mainline up to div_idx-1
        full_prefix = rel_prefix
        fixed_queue.append((div_idx, full_prefix, subtree))
    var_queue = fixed_queue

    emitted_blocks: List[str] = []
    # To avoid runaway, cap number of blocks (for weird positions)
    block_cap = 500
    while var_queue and block_cap > 0:
        block_cap -= 1
        div_idx, prefix_moves, subtree = var_queue.pop(0)
        emitted_blocks.append(f"変化：{div_idx}手")
        branch_lines, nested = emit_lines_for_branch(board0, prefix_moves, subtree, start_idx=div_idx, sec_per_move=sec_per_move)
        emitted_blocks.extend(branch_lines)
        # add terminal
        # compute last move number in this branch block
        last_idx = div_idx + len(branch_lines)
        emitted_blocks.append(f"{last_idx:4d} 詰み         ( 0:{sec_per_move:02d}/00:00:{sec_per_move:02d})")
        # queue nested
        for n in nested:
            var_queue.append(n)

    text = "\n".join(header + lines + emitted_blocks) + "\n"
    with open(outfile, "wb") as f:
        f.write(text.encode("cp932", errors="replace"))


def build_mainline_and_variations(board, tree: SolveNode, start_idx: int = 1):
    """
    Produce:
      main_moves: list of shogi.Move along first-child path
      variations: list of (div_idx, prefix_moves, branch_root_move, branch_subtree)
    We'll later output each variation as '変化：{div_idx}手' and its line sequence.
    """
    main_moves: List[object] = []
    variations: List[Tuple[int, List[object], SolveNode]] = []  # (div_idx, prefix_moves, subtree_node_with_move_at_root)

    def walk(node: SolveNode, prefix: List[object], idx: int):
        if not node.children:
            return
        # children are SolveNode(move=mv, children=...)
        # choose first as mainline
        kids = node.children
        main = kids[0]
        # extra branches become variations
        for extra in kids[1:]:
            # variation subtree starts with extra move at current idx
            variations.append((idx, prefix.copy(), extra))
        # advance mainline
        main_moves.append(main.move)
        prefix.append(main.move)
        walk(main, prefix, idx+1)
        prefix.pop()

    walk(tree, [], start_idx)
    return main_moves, variations


def emit_lines_for_branch(board0, prefix_moves: List[object], branch: SolveNode, start_idx: int, sec_per_move: int = 3):
    """
    Emit kif lines for a branch:
      - apply prefix_moves to board0 to reach divergence parent
      - then apply branch.move and follow first-child mainline
      - collect further nested variations inside this branch and return them
    """
    board = copy.deepcopy(board0)
    prev_to = None
    total_sec = 0
    idx = 1

    # play prefix to set prev_to and idx counters to divergence point
    for mv in prefix_moves:
        line, prev_to = kif_line_for_shogi_move(idx, board, mv, prev_to, sec_per_move, total_sec + sec_per_move)
        total_sec += sec_per_move
        board.push(mv)
        idx += 1

    # now divergence starts at start_idx == len(prefix_moves)+1 typically
    # but we will output from start_idx onward only (KIF "変化" blocks assume move numbers restart at start_idx)
    # So: reset idx to start_idx and total_sec to cumulative at that ply.
    idx = start_idx
    # prev_to should be to-square of move idx-1, already correct.

    lines: List[str] = []
    nested_variations: List[Tuple[int, List[object], SolveNode]] = []

    # Build a temporary tree rooted at this branch
    temp_root = SolveNode(move=None, children=[branch])

    # mainline for this branch (first-child path)
    main_moves, variations = build_mainline_and_variations(board, temp_root, start_idx=idx)

    # Emit main_moves (includes branch.move as first element)
    for mv in main_moves:
        total_sec += sec_per_move
        line, prev_to = kif_line_for_shogi_move(idx, board, mv, prev_to, sec_per_move, total_sec)
        lines.append(line)
        board.push(mv)
        idx += 1

    # Collect variations; each variation tuple gives (div_idx, prefix_moves_in_tree, subtree)
    # Need to transform prefix inside temp_root (relative) into full prefix list.
    for div_idx, rel_prefix, subtree in variations:
        full_prefix = prefix_moves + rel_prefix
        nested_variations.append((div_idx, full_prefix, subtree))

    return lines, nested_variations
