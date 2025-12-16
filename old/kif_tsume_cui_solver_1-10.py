#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
kif_tsume_cui_ankif_v4.py

Tsume-shogi focused CUI tool:
- Create start position in CUI (board + hands + side-to-move)
- Fast numeric input for moves (move / promote / drop-pick)
- KIF output tuned to "Kifu for Mac V0.53" style (ANKIF-friendly)
- Optional python-shogi integration (recommended):
    pip install python-shogi
  Enables:
    * strict legality / checkmate detection
    * solve command: enumerate forced mate lines up to N plies (odd number, e.g., 1/3/5/7/9)

Notes:
- We intentionally focus on tsume usage. "Hirate" etc. can be added later as separate modes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import json
import pathlib
import datetime
import time
import hashlib

from pathlib import Path

def _output_dir() -> Path:
    # OUTPUT directory next to this script
    return Path(__file__).resolve().parent / "OUTPUT"

def _ensure_output_dir() -> Path:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out

def _input_dir() -> Path:
    # INPUT directory next to this script
    return Path(__file__).resolve().parent / "INPUT"

def _ensure_input_dir() -> Path:
    inp = _input_dir()
    inp.mkdir(parents=True, exist_ok=True)
    return inp

def _resolve_kif_path(name: str) -> Path:
    """If name is relative (no directory), store into OUTPUT/."""
    p = Path(name)
    if p.is_absolute() or p.parent != Path('.'):
        return p
    return _ensure_output_dir() / p.name

def _resolve_existing_kif_path(name: str) -> Path:
    """Resolve a KIF path for reading. If relative and not found, try OUTPUT/."""
    p = Path(name)
    if p.exists():
        return p
    if not p.is_absolute() and p.parent == Path('.'):
        q = _output_dir() / p.name
        if q.exists():
            return q
    return p

import re

# -------- Optional library: python-shogi --------
try:
    import shogi  # pip install python-shogi  (import name is "shogi")
    HAS_PYSHOGI = True
except Exception:
    shogi = None
    HAS_PYSHOGI = False

# ----------------- constants -----------------

FW_DIGITS = {str(i): ch for i, ch in enumerate("０１２３４５６７８９")}
RANK_KANJI = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}

PIECE_JP = {"P":"歩","L":"香","N":"桂","S":"銀","G":"金","B":"角","R":"飛","K":"玉"}
PROMOTED_JP = {"P":"と","L":"成香","N":"成桂","S":"成銀","B":"馬","R":"竜"}
PROMOTABLE = set(["P","L","N","S","B","R"])

# Total piece counts in standard shogi (excluding kings in "hands" output)
TOTAL_COUNTS = {"R":2,"B":2,"G":4,"S":4,"N":4,"L":4,"P":18,"K":2}

# piyo-like board display tokens
KIND_TO_PYO = {
    ("P",False):"歩", ("L",False):"香", ("N",False):"桂", ("S",False):"銀", ("G",False):"金",
    ("B",False):"角", ("R",False):"飛", ("K",False):"玉",
    ("P",True):"と", ("L",True):"成香", ("N",True):"成桂", ("S",True):"成銀",
    ("B",True):"馬", ("R",True):"竜",
}

# ----------------- helpers -----------------

def format_total_time(total_sec: int) -> str:
    """秒数を HH:MM:SS に整形（KIFの消費時間表示用）"""
    if total_sec < 0:
        total_sec = 0
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def now_yyyy_mm_dd_hhmmss() -> str:
    # 015.kif style: YYYY/MM/DD HH:MM:SS
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def sq_to_kif(file_: int, rank: int) -> str:
    return f"{FW_DIGITS[str(file_)]}{RANK_KANJI[rank]}"

def sq_to_paren(file_: int, rank: int) -> str:
    return f"({file_}{rank})"

def parse_piece_token(tok: str) -> Optional["Piece"]:
    """
    tok examples:
      P, R, K
      vP, vR, vK    (gote)
      +P, +B, +R    (promoted)
      v+P, v+R      (gote promoted)
      .             (empty)
    """
    tok = tok.strip()
    if tok == ".":
        return None
    color = "B"
    prom = False
    if tok.startswith("v"):
        color = "W"
        tok = tok[1:]
    if tok.startswith("+"):
        prom = True
        tok = tok[1:]
    tok = tok.upper()
    if tok not in PIECE_JP:
        raise ValueError("駒指定は P L N S G B R K（後手はv、成りは+、空は. 例: v+R）")
    return Piece(color, tok, prom)

def build_piece_menu(prefer_side: str = "B") -> List[Tuple[str,str]]:
    """
    Return list of (token, label).
    prefer_side: "B" => show ▲ first, "W" => show △ first
    """
    base = [("P","歩"),("L","香"),("N","桂"),("S","銀"),("G","金"),("B","角"),("R","飛"),("K","玉")]
    promo = [("P","と"),("L","成香"),("N","成桂"),("S","成銀"),("B","馬"),("R","竜")]

    def mk_entries(color: str, items, promoted: bool):
        entries = []
        for k, jp in items:
            tok = k
            if promoted:
                tok = "+" + tok
            if color == "W":
                tok = "v" + tok
                label = "△" + jp
            else:
                label = "▲" + jp
            entries.append((tok, label))
        return entries

    first = "B" if prefer_side == "B" else "W"
    second = "W" if first == "B" else "B"

    menu: List[Tuple[str,str]] = []
    menu.append((".", "・(消す)"))
    menu += mk_entries(first, base, False)
    menu += mk_entries(second, base, False)
    menu += mk_entries(first, promo, True)
    menu += mk_entries(second, promo, True)
    return menu

def inv_count_kanji(n: int) -> str:
    inv = {
        1:"",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",
        10:"十",11:"十一",12:"十二",13:"十三",14:"十四",15:"十五",16:"十六",17:"十七",18:"十八"
    }
    return inv.get(n, str(n))

# ----------------- model -----------------

@dataclass
class Piece:
    color: str  # "B" (sente) or "W" (gote)
    kind: str   # "P,L,N,S,G,B,R,K"
    prom: bool = False

    def display_name(self) -> str:
        if self.prom and self.kind in PROMOTED_JP:
            return PROMOTED_JP[self.kind]
        return PIECE_JP[self.kind]

@dataclass
class Move:
    is_drop: bool
    kind: str
    from_sq: Optional[Tuple[int,int]]
    to_sq: Tuple[int,int]
    promote: bool
    same_as_prev: bool

class ShogiPosition:
    def __init__(self):
        self.board: Dict[Tuple[int,int], Optional[Piece]] = {(f,r):None for f in range(1,10) for r in range(1,10)}
        self.hands: Dict[str, Dict[str, int]] = {"B":{}, "W":{}}
        self.side_to_move: str = "B"
        self.moves: List[Move] = []
        self._history: List[Tuple[Dict, Dict, str, List[Move]]] = []

    def clone_state(self):
        return (copy.deepcopy(self.board), copy.deepcopy(self.hands), self.side_to_move, copy.deepcopy(self.moves))

    def push_history(self):
        self._history.append(self.clone_state())

    def undo(self) -> bool:
        if not self._history:
            return False
        self.board, self.hands, self.side_to_move, self.moves = self._history.pop()
        return True

    def clear_moves(self):
        self.moves = []
        self._history = []

    def clear_all(self):
        self.board = {(f,r):None for f in range(1,10) for r in range(1,10)}
        self.hands = {"B":{}, "W":{}}
        self.side_to_move = "B"
        self.clear_moves()

    # ---- setup editing ----
    def set_piece(self, sq: Tuple[int,int], p: Optional[Piece]):
        self.board[sq] = p

    def set_hand(self, color: str, kind: str, n: int):
        if n <= 0:
            self.hands[color].pop(kind, None)
        else:
            self.hands[color][kind] = n

    # ---- numeric input ----
    def parse_numeric(self, s: str):
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
        f1,r1,f2,r2 = map(int, core)
        for x in (f1,r1,f2,r2):
            if not (1 <= x <= 9):
                raise ValueError("マスは 11〜99 の範囲です")
        return ("move", None, (f1,r1), (f2,r2), promote)

    # ---- minimal legality (for editing convenience) ----
    # We intentionally keep it permissive; strict check is via python-shogi when available.

    def _remove_from_hand(self, color: str, kind: str):
        c = self.hands[color].get(kind, 0)
        if c <= 0:
            raise ValueError("持ち駒がありません")
        if c == 1:
            del self.hands[color][kind]
        else:
            self.hands[color][kind] = c - 1

    def _add_to_hand(self, color: str, kind: str):
        self.hands[color][kind] = self.hands[color].get(kind, 0) + 1

    def drop_candidates(self, to: Tuple[int,int]) -> List[str]:
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
                for r in range(1,10):
                    p = self.board[(file_, r)]
                    if p and p.color == self.side_to_move and p.kind == "P" and not p.prom:
                        nifu = True
                        break
                if nifu:
                    continue
            result.append(k)
        return result

    def apply_move_minimal(self, kind: str, frm: Optional[Tuple[int,int]], to: Tuple[int,int], promote: bool, is_drop: bool) -> Move:
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
    def hands_to_piyo(self, color: str) -> str:
        order = ["R","B","G","S","N","L","P"]
        parts = []
        for k in order:
            n = self.hands[color].get(k, 0)
            if n <= 0:
                continue
            parts.append(PIECE_JP[k] + inv_count_kanji(n))
        return " ".join(parts) + (" " if parts else "")

    def board_to_piyo(self) -> str:
        lines = []
        lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
        lines.append("+---------------------------+")
        for r in range(1,10):
            row = []
            for f in range(9,0,-1):
                p = self.board[(f,r)]
                if p is None:
                    row.append(" ・")
                else:
                    name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
                    cell = ("v" + name) if p.color == "W" else (" " + name)
                    row.append(cell)
            lines.append("|" + "".join(row) + f"|{RANK_KANJI[r]}")
        lines.append("+---------------------------+")
        return "\n".join(lines)

# ----------------- KIF formatting (for python-shogi moves) -----------------

def _sq_from_shogi_square(sq: int) -> Tuple[int,int]:
    # python-shogi square: 0..80. 0 is 9a (file 9, rank 1), 8 is 1a, 72 is 9i, 80 is 1i.
    file_ = 9 - (sq % 9)
    rank_ = (sq // 9) + 1
    return file_, rank_

def _piece_type_to_kind(pt: int) -> str:
    # Map python-shogi piece type to our kind letter (base kind for promoted pieces too)
    mapping = {
        getattr(shogi, "PAWN", None): "P",
        getattr(shogi, "LANCE", None): "L",
        getattr(shogi, "KNIGHT", None): "N",
        getattr(shogi, "SILVER", None): "S",
        getattr(shogi, "GOLD", None): "G",
        getattr(shogi, "BISHOP", None): "B",
        getattr(shogi, "ROOK", None): "R",
        getattr(shogi, "KING", None): "K",

        # promoted piece types (python-shogi uses separate constants)
        getattr(shogi, "PRO_PAWN", None): "P",
        getattr(shogi, "PRO_LANCE", None): "L",
        getattr(shogi, "PRO_KNIGHT", None): "N",
        getattr(shogi, "PRO_SILVER", None): "S",
        getattr(shogi, "HORSE", None): "B",   # 馬
        getattr(shogi, "DRAGON", None): "R",  # 竜
    }
    # Remove None keys (when a constant doesn't exist in a given python-shogi version)
    mapping = {k:v for k,v in mapping.items() if k is not None}
    return mapping.get(pt, "P")

def _is_promoted_piece_type(pt: int) -> bool:
    promoted = {
        getattr(shogi, "PRO_PAWN", None),
        getattr(shogi, "PRO_LANCE", None),
        getattr(shogi, "PRO_KNIGHT", None),
        getattr(shogi, "PRO_SILVER", None),
        getattr(shogi, "HORSE", None),
        getattr(shogi, "DRAGON", None),
    }
    return pt in {x for x in promoted if x is not None}


def kif_line_for_shogi_move(idx: int, board_before, move, prev_to: Optional[Tuple[int,int]], sec: int, total_sec: int) -> Tuple[str, Tuple[int,int]]:
    """
    Returns (line, to_sq_file_rank) using 015.kif-like time field:
       ( 0:01/00:00:01)
    """
    to_file, to_rank = _sq_from_shogi_square(move.to_square)
    dst = "同" if (prev_to is not None and prev_to == (to_file,to_rank)) else sq_to_kif(to_file, to_rank)

    if move.drop_piece_type is not None:
        kind = _piece_type_to_kind(move.drop_piece_type)
        body = f"{dst}{PIECE_JP[kind]}打"
        from_part = ""
    else:
        # moving piece pre-move
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
        from_part = sq_to_paren(fr_file, fr_rank)
        body = f"{dst}{name}{from_part}"

    # spacing: align similar to sample (not strict)
    # We'll pad to 13 columns after body start.
    time_part = f"( 0:{sec:02d}/00:00:{total_sec:02d})"
    line = f"{idx:4d} {body:<12} {time_part}"
    return line, (to_file,to_rank)

# ----------------- SFEN building & gote-remaining hands -----------------

def compute_gote_remaining(board_map: Dict[Tuple[int,int], Optional[Piece]], sente_hand: Dict[str,int]) -> Dict[str,int]:
    """
    "後手の持駒" is the remaining pieces not present on board and not in sente hand.
    - Count promoted pieces as their base kind.
    - Exclude kings from hand output.
    """
    used = {k: 0 for k in TOTAL_COUNTS.keys()}
    # board pieces
    for sq, p in board_map.items():
        if p is None:
            continue
        kind = p.kind
        # promoted counts as base kind
        if kind in used:
            used[kind] += 1
    # sente hand
    for kind, n in sente_hand.items():
        if kind in used:
            used[kind] += n

    remaining: Dict[str,int] = {}
    for kind, total in TOTAL_COUNTS.items():
        if kind == "K":
            continue
        left = total - used.get(kind, 0)
        if left > 0:
            remaining[kind] = left
    return remaining

def hands_to_sfen(hands_b: Dict[str,int], hands_w: Dict[str,int]) -> str:
    # SFEN hand format: pieces with optional count prefix, uppercase for black, lowercase for white
    order = ["R","B","G","S","N","L","P"]
    parts: List[str] = []
    def add(kind: str, n: int, is_black: bool):
        if n <= 0:
            return
        c = kind if is_black else kind.lower()
        if n == 1:
            parts.append(c)
        else:
            parts.append(f"{n}{c}")
    for k in order:
        add(k, hands_b.get(k,0), True)
    for k in order:
        add(k, hands_w.get(k,0), False)
    return "-" if not parts else "".join(parts)

def board_to_sfen(board_map: Dict[Tuple[int,int], Optional[Piece]]) -> str:
    # ranks 1..9 correspond to a..i (top to bottom)
    rows = []
    for r in range(1,10):
        empties = 0
        row = ""
        for f in range(9,0,-1):
            p = board_map[(f,r)]
            if p is None:
                empties += 1
                continue
            if empties:
                row += str(empties)
                empties = 0
            ch = p.kind
            if p.prom:
                row += "+"
            row += ch if p.color == "B" else ch.lower()
        if empties:
            row += str(empties)
        rows.append(row)
    return "/".join(rows)

def snapshot_to_sfen(board_map, hands_b, side_to_move: str, gote_hands_auto: Dict[str,int]) -> str:
    board_part = board_to_sfen(board_map)
    turn = "b" if side_to_move == "B" else "w"
    hands_part = hands_to_sfen(hands_b, gote_hands_auto)
    return f"{board_part} {turn} {hands_part} 1"


# ----------------- Start-position save/load & KIF preview -----------------

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

def save_startpos_file(filename: str, board_map, hands_b, side_to_move: str, sente: str, gote: str):
    gote_auto = compute_gote_remaining(board_map, hands_b)
    sfen = snapshot_to_sfen(board_map, hands_b, side_to_move, gote_auto)
    data = {
        "format": "kif_tsume_cui_startpos_v1",
        "sfen": sfen,
        "sente": sente,
        "gote": gote,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_startpos_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    sfen = data.get("sfen")
    if not sfen:
        raise ValueError("startposファイルにsfenがありません")
    board_map, hands_b, side_to_move = sfen_to_snapshot(sfen)
    sente = data.get("sente")
    gote = data.get("gote")
    return board_map, hands_b, side_to_move, sente, gote

def preview_kif_file(filename: str, max_lines: int = 120):
    p = _resolve_existing_kif_path(filename)
    try:
        raw = pathlib.Path(p).read_bytes()
        txt = raw.decode("cp932", errors="replace")
    except Exception:
        txt = pathlib.Path(filename).read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    n = min(len(lines), max_lines)
    print(f"[preview] {p}（全{len(lines)}行） 先頭{n}行を表示します")
    for i in range(n):
        print(lines[i])
    if len(lines) > n:
        print(f"...（残り {len(lines)-n} 行）")


# ----------------- Batch import/split from existing KIF -----------------

_KIF_ROW_LABELS = ["一","二","三","四","五","六","七","八","九"]

_PYO_PIECES_LONGEST = [
    "成香","成桂","成銀",  # 2-char
    "馬","竜","と",        # 1-char promoted (note: 竜 may appear)
    "玉","飛","角","金","銀","桂","香","歩",
]

_PYO_TO_KIND_PROM = {
    "歩": ("P", False),
    "香": ("L", False),
    "桂": ("N", False),
    "銀": ("S", False),
    "金": ("G", False),
    "角": ("B", False),
    "飛": ("R", False),
    "玉": ("K", False),
    "と": ("P", True),
    "成香": ("L", True),
    "成桂": ("N", True),
    "成銀": ("S", True),
    "馬": ("B", True),
    "竜": ("R", True),
}

_KANJI_DIGITS = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
def _kanji_count_to_int(s: str) -> int:
    """Parse counts like '二','四','十','十七' used by our KIF output."""
    s = s.strip()
    if not s:
        return 1
    if s == "十":
        return 10
    if "十" in s:
        a, b = s.split("十", 1)
        tens = _KANJI_DIGITS.get(a, 1) if a else 1
        ones = _KANJI_DIGITS.get(b, 0) if b else 0
        return tens * 10 + ones
    return _KANJI_DIGITS.get(s, 1)

def _parse_hands_line_piyo(line: str) -> Dict[str,int]:
    """
    Parse: '先手の持駒：金' or '後手の持駒：飛二 角二 ...'
    Returns dict kind->count (K excluded).
    """
    if "：" not in line:
        return {}
    _, rest = line.split("：", 1)
    rest = rest.strip()
    if not rest or rest == "なし":
        return {}
    out: Dict[str,int] = {}
    # tokens separated by spaces (our output)
    for tok in rest.split():
        # tok like "飛二" or "歩十七" or "金"
        name = None
        cnt = 1
        # find piece name by matching known names
        for pn in _PYO_PIECES_LONGEST:
            if tok.startswith(pn):
                name = pn
                suffix = tok[len(pn):]
                if suffix:
                    cnt = _kanji_count_to_int(suffix)
                break
        if not name:
            continue
        kind, _prom = _PYO_TO_KIND_PROM.get(name, (None, None))
        if not kind or kind == "K":
            continue
        out[kind] = out.get(kind, 0) + int(cnt)
    return out

def _consume_one_cell(s: str) -> Tuple[Optional[Tuple[str,bool,str]], str]:
    """
    Consume one board cell from our piyo diagram row string.
    Returns (kind, prom, color) or None, and remaining string.
    Our cell format when generated:
      - empty: ' ・'
      - sente piece: ' 歩' or ' 成香' etc (leading space)
      - gote piece:  'v歩' or 'v成香' etc (no leading space, starts with v)
    """
    if not s:
        return None, ""
    color = None
    if s[0] == " ":
        s = s[1:]
        color = "B"
    elif s[0] == "v":
        s = s[1:]
        color = "W"
    else:
        # defensive: skip one char
        return None, s[1:]

    if not s:
        return None, ""
    if s.startswith("・"):
        return None, s[1:]

    # match longest piece name
    name = None
    for pn in _PYO_PIECES_LONGEST:
        if s.startswith(pn):
            name = pn
            s = s[len(pn):]
            break
    if not name:
        # unknown token; skip 1 char
        return None, s[1:]
    kind, prom = _PYO_TO_KIND_PROM.get(name, (None, None))
    if not kind:
        return None, s
    return (kind, bool(prom), color), s

def parse_kif_startpos_piyo(text: str):
    """
    Parse start position from our own KIF output (piyo diagram + hands lines).
    Returns (board_map, hands_b, side_to_move).
    """
    lines = text.splitlines()

    hands_b: Dict[str,int] = {}
    side_to_move = "B"

    # hands
    for ln in lines:
        if ln.startswith("先手の持駒："):
            hands_b = _parse_hands_line_piyo(ln)
            break

    # board diagram block
    # find the header line containing digits and the first border line
    i0 = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("９") and "８" in ln and "１" in ln:
            # next line should be +--- border
            if i+1 < len(lines) and lines[i+1].startswith("+"):
                i0 = i+2
                break
    if i0 is None:
        raise ValueError("盤面図が見つかりません（このKIF形式は未対応の可能性）")

    board_map: Dict[Tuple[int,int], Optional[Piece]] = {(f,r): None for f in range(1,10) for r in range(1,10)}
    for r in range(1,10):
        ln = lines[i0 + (r-1)]
        # like: | ・ ・ ・ ・ ・v玉 ・ ・ ・|二
        if "|" not in ln:
            continue
        body = ln.split("|",2)[1]
        body = body.rstrip()
        # parse 9 cells from left (file 9) to right (file 1)
        rest = body
        for col in range(9):
            cell, rest = _consume_one_cell(rest)
            file_ = 9 - col
            rank_ = r
            if cell is None:
                continue
            kind, prom, color = cell
            board_map[(file_, rank_)] = Piece(color=color, kind=kind, prom=prom)
    return board_map, hands_b, side_to_move

def parse_kif_moves_and_variations(text: str):
    """
    Very small parser:
    - Extract main moves section until '変化：' or EOF
    - Extract each variation as list of move-lines (strings) with their start ply
    Returns: (main_move_lines, variations[(start_ply, move_lines)], ply_limit_guess)
    """
    lines = text.splitlines()
    # find moves header
    idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("手数----指手"):
            idx = i+1
            break
    if idx is None:
        return [], [], 0

    main = []
    vars_ = []
    cur_var = None  # (start_ply, list)
    for ln in lines[idx:]:
        if ln.startswith("変化："):
            # close current var
            if cur_var is not None:
                vars_.append(cur_var)
            m = re.search(r"変化：\s*(\d+)手", ln)
            sp = int(m.group(1)) if m else 1
            cur_var = (sp, [])
            continue
        # end markers
        if ln.strip() == "":
            continue
        if cur_var is not None:
            cur_var[1].append(ln)
        else:
            # include until a '変化：' appears
            main.append(ln)
    if cur_var is not None:
        vars_.append(cur_var)

    # guess ply limit from main: count moves until '詰み'/'投了'
    ply = 0
    for ln in main:
        if "詰み" in ln or "投了" in ln:
            break
        m = re.match(r"\s*(\d+)\s", ln)
        if m:
            ply = max(ply, int(m.group(1)))
    return main, vars_, ply

def _ensure_output_dir() -> Path:
    outdir = _output_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _basename_no_ext(p: str) -> str:
    return pathlib.Path(p).stem

def batch_process_kif(path: str, default_ply: Optional[int] = None,
                      limits: Optional[SolveLimits] = None,
                      check_only: bool = True) -> List[str]:
    """
    Process one kif:
    - If it contains '変化：' blocks: split each branch (main + each variation) into separate files.
      (We do not attempt to re-solve; we just split text-based branches by reusing our own writer.)
    - If it has no variations: try to parse startpos, then solve for mates within ply (from file or default_ply),
      and write each mate line as separate KIF.
      If no mate found within ply: write the original file as OUTPUT/<base>.kif.
    Returns written file paths.
    """
    outdir = _ensure_output_dir()
    raw = pathlib.Path(path).read_bytes()
    try:
        txt = raw.decode("cp932", errors="replace")
    except Exception:
        txt = raw.decode("utf-8", errors="replace")

    main_lines, variations, ply_guess = parse_kif_moves_and_variations(txt)
    base = _basename_no_ext(pathlib.Path(path).name)

    written: List[str] = []

    # Case A: has variations -> split (best-effort; use original text blocks)
    if variations:
        # Write main as _001, then each variation as _002... by reusing original text but stripping other variations
        # This preserves maximum ANKIF compatibility because it's basically the same format.
        def _write_text(outname: str, body_lines: List[str]) -> str:
            # keep header up to moves header line, then body_lines, then end
            lines = txt.splitlines()
            # keep everything until moves header (inclusive)
            keep = []
            for ln in lines:
                keep.append(ln)
                if ln.strip().startswith("手数----指手"):
                    break
            out_txt = "\n".join(keep + body_lines) + "\n"
            p = outdir / outname
            p.write_bytes(out_txt.encode("cp932", errors="replace"))
            return str(p)

        written.append(_write_text(f"{base}.kif", main_lines))
        for j, (sp, mvlines) in enumerate(variations, start=2):
            # prepend main up to sp-1 from main_lines (text-based; assume numbering matches)
            prefix = []
            for ln in main_lines:
                m = re.match(r"\s*(\d+)\s", ln)
                if m and int(m.group(1)) >= sp:
                    break
                prefix.append(ln)
            combined = prefix + mvlines
            written.append(_write_text(f"{base}_{j:03d}.kif", combined))
        return written

    # Case B: no variations -> solve
    try:
        board_map, hands_b, stm = parse_kif_startpos_piyo(txt)
    except Exception as e:
        # cannot parse -> just copy original
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: 開始局面の解析に失敗（{e}）。元KIFをコピーしました: {dst}")
        return [str(dst)]

    # Determine ply
    ply = default_ply if default_ply is not None else (ply_guess if ply_guess else 9)
    if ply % 2 == 0:
        ply += 1  # tsume typically odd for attacker to finish

    try:
        import shogi as pyshogi
    except Exception:
        # no python-shogi -> just copy
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: python-shogi がないため解探索はスキップ。元KIFをコピー: {dst}")
        return [str(dst)]

    gote_auto = compute_gote_remaining(board_map, hands_b)
    sfen = snapshot_to_sfen(board_map, hands_b, stm, gote_auto)
    board0 = pyshogi.Board(sfen)

    stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}
    tree = solve_mate_tree(board0, ply, attacker_turn=pyshogi.BLACK, check_only=check_only, stats=stats, limits=limits)
    elapsed = time.perf_counter() - float(stats.get("start", 0.0))
    if tree is None or not tree.children:
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: {ply}手以内の詰み筋が見つからず（nodes={stats['nodes']} {elapsed:.3f}s）。元KIFをコピー: {dst}")
        return [str(dst)]

    # produce all solution move-paths from tree
    tree = prune_tree_to_max_leaves(tree, limits.max_solutions if limits else 300)
    paths = enumerate_solution_paths(tree)
    if not paths:
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: 解答筋が空でした（nodes={stats['nodes']} {elapsed:.3f}s）。元KIFをコピー: {dst}")
        return [str(dst)]

    # Write: first solution -> base.kif, others -> base_002.kif ...
    written: List[str] = []
    seen_kif: Dict[str,str] = {}
    p0 = generate_kif_single_line(board0, hands_b, gote_auto, "", "", paths[0], f"{base}.kif", seen_kif)
    if p0:
        written.append(str(p0))
    for j, mvlist in enumerate(paths[1:], start=2):
        fname = f"{base}_{j:03d}.kif"
        pw = generate_kif_single_line(board0, hands_b, gote_auto, "", "", mvlist, fname, seen_kif)
        if pw:
            written.append(str(pw))

    cutoff_note = "（制限で打ち切り・部分解の可能性）" if stats.get("cutoff") else ""
    print(f"[batch] {path}: 解答筋={len(paths)} {cutoff_note} / 探索={elapsed:.3f}s nodes={stats['nodes']} -> OUTPUT")
    return written

def batch_process_path(path: str, default_ply: Optional[int] = None,
                       limits: Optional[SolveLimits] = None,
                       check_only: bool = True) -> List[str]:
    """
    Process a folder or one file.
      batch <dir> [ply]
      batch <file.kif> [ply]
    """
    p = pathlib.Path(path)
    written_all: List[str] = []
    if p.is_dir():
        for kif in sorted(p.glob("*.kif")):
            written_all.extend(batch_process_kif(str(kif), default_ply=default_ply, limits=limits, check_only=check_only))
    else:
        written_all.extend(batch_process_kif(str(p), default_ply=default_ply, limits=limits, check_only=check_only))
    return written_all
# ----------------- Solve (python-shogi) -----------------

@dataclass
class SolveNode:
    move: Optional[object]  # shogi.Move or None for root
    children: List["SolveNode"]

def _is_attacker_turn(board, attacker_turn: int) -> bool:
    return board.turn == attacker_turn


@dataclass
class SolveLimits:
    max_nodes: int = 50000
    max_time_sec: float = 5.0
    max_solutions: int = 300

def solve_mate_tree(board, ply_left: int, attacker_turn: int, check_only: bool = True,
                    memo: Optional[Dict[Tuple[str,int], Optional[SolveNode]]] = None,
                    stats: Optional[Dict[str, object]] = None,
                    limits: Optional[SolveLimits] = None) -> Optional[SolveNode]:
    """
    Returns a SolveNode tree if attacker has a forced mate within ply_left plies, else None.
    The tree includes:
      - attacker nodes: all winning moves (OR)
      - defender nodes: all legal replies (AND), but only if all replies still lead to mate.
    """
    if memo is None:
        memo = {}
    if stats is None:
        stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}
    if limits is None:
        limits = SolveLimits()

    stats["nodes"] = int(stats.get("nodes", 0)) + 1
    # time/node cutoff
    if stats["nodes"] > limits.max_nodes:
        stats["cutoff"] = True
        return None
    if limits.max_time_sec is not None:
        if (time.perf_counter() - float(stats.get("start", 0.0))) > float(limits.max_time_sec):
            stats["cutoff"] = True
            return None

    key = (board.sfen(), ply_left)
    if key in memo:
        return memo[key]

    if ply_left <= 0:
        memo[key] = None
        return None

    # If side to move is already checkmated, it means the previous move delivered mate.
    if board.is_checkmate():
        # This position is terminal from the perspective of the player who just moved.
        # But in our recursion we handle mate right after push, so treat as success stopper.
        memo[key] = SolveNode(move=None, children=[])
        return memo[key]

    attacker_to_move = _is_attacker_turn(board, attacker_turn)

    legal = list(board.legal_moves)

    # Optional pruning: in tsume, attacker moves are usually checks only.
    if attacker_to_move and check_only:
        filtered = []
        for mv in legal:
            board.push(mv)
            ok = board.is_check()  # opponent is in check now
            board.pop()
            if ok:
                filtered.append(mv)
        legal = filtered

    if attacker_to_move:
        winning_children: List[SolveNode] = []
        for mv in legal:
            board.push(mv)
            if board.is_checkmate():
                # mate delivered
                stats["solutions"] = int(stats.get("solutions", 0)) + 1
                if stats["solutions"] > limits.max_solutions:
                    stats["cutoff"] = True
                    board.pop()
                    break
                winning_children.append(SolveNode(move=mv, children=[]))
                board.pop()
                continue
            child = solve_mate_tree(board, ply_left - 1, attacker_turn, check_only, memo, stats, limits)
            board.pop()
            if child is not None:
                winning_children.append(SolveNode(move=mv, children=child.children))
        node = SolveNode(move=None, children=winning_children) if winning_children else None
        memo[key] = node
        return node
    else:
        # Defender to move: attacker must survive all replies
        defender_children: List[SolveNode] = []
        for mv in legal:
            board.push(mv)
            # If defender mates attacker (rare in tsume), treat as escape (attacker failed)
            if board.is_checkmate():
                board.pop()
                memo[key] = None
                return None
            child = solve_mate_tree(board, ply_left - 1, attacker_turn, check_only, memo, stats, limits)
            board.pop()
            if child is None:
                memo[key] = None
                return None
            defender_children.append(SolveNode(move=mv, children=child.children))
        node = SolveNode(move=None, children=defender_children)
        memo[key] = node
        return node

def count_solutions(tree: SolveNode) -> int:
    # Count distinct leaf mate lines in the tree.
    def rec(node: SolveNode) -> int:
        if not node.children:
            return 1
        return sum(rec(ch) for ch in node.children)
    return rec(tree)

def prune_tree_to_max_leaves(tree: SolveNode, max_leaves: int) -> SolveNode:
    """Prune SolveNode tree in-place (copy) so that total leaf lines <= max_leaves."""
    if max_leaves is None or max_leaves <= 0:
        return tree
    tree = copy.deepcopy(tree)

    def rec(node: SolveNode, remaining: List[int]):
        if remaining[0] <= 0:
            node.children = []
            return
        if not node.children:
            remaining[0] -= 1
            return
        new_children = []
        for ch in node.children:
            if remaining[0] <= 0:
                break
            rec(ch, remaining)
            new_children.append(ch)
        node.children = new_children

    rec(tree, [max_leaves])
    return tree


# ----------------- KIF with variations from solve tree -----------------

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


def enumerate_solution_paths(tree: SolveNode) -> List[List[object]]:
    """Return all move-sequences (shogi.Move list) from root to each leaf."""
    paths: List[List[object]] = []
    def rec(node: SolveNode, cur: List[object]):
        if node.move is not None:
            cur = cur + [node.move]
        if not node.children:
            paths.append(cur)
            return
        for ch in node.children:
            rec(ch, cur)
    rec(tree, [])
    return paths

def generate_kif_single_line(board0, hands_b: Dict[str,int], gote_hands_auto: Dict[str,int],
                             sente_name: str, gote_name: str, moves: List[object], outfile: str,
                             seen: Optional[Dict[str,str]] = None) -> Optional[Path]:
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
            prev = Path(seen[key]).name
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

# Helpers for header board/hand lines in KIF

def _hands_to_line(hands: Dict[str,int]) -> str:
    order = ["R","B","G","S","N","L","P"]
    parts = []
    for k in order:
        n = hands.get(k,0)
        if n <= 0:
            continue
        parts.append(f"{PIECE_JP[k]}{inv_count_kanji(n)}")
    return " ".join(parts) + (" " if parts else "")

def board0_to_piyo(board0) -> str:
    # board0 is python-shogi Board
    lines = []
    lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
    lines.append("+---------------------------+")
    for r in range(1,10):
        row = []
        for f in range(9,0,-1):
            # convert to square index
            sq = (r-1)*9 + (9-f)
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

# ----------------- Help -----------------

HELP_MAIN = """\
============================================================
  詰将棋用 KIF 入力 CUIツール（v4）
============================================================

【最短の流れ】
  1) 局面作成: show → p / h / turn → start
  2) 手順入力: 7776 / 22331 / 076(打ち)
  3) 保存:     s out.kif

【発展：自動で全詰み筋を探索（python-shogi必要）】
  solve 9 out.kif
    - 開始局面（startで確定）から、9手詰（=9プライ）以内の
      「先手の強制詰み筋」を探索して、分岐付きKIFに出力します。
    - 先手の手は基本「王手になる手」に限定（詰将棋向け高速化）

【コマンド】
  show                 : 盤面表示
  p 55                 : 駒配置（メニュー）
  p 55 v+R / p 55 .     : 駒配置（直指定 / 消す）
  h b P 2 / h w R 1     : 先手/後手の持ち駒（※後手は出力時に自動残駒化も可能）
  turn b / turn w      : 手番
  start                : 開始局面を確定（ここから手順入力）
  end                  : 終局を選択（詰み/投了） ※省略可（詰み自動判定あり）
  7776 / 22331         : 移動（4桁）/ 成り（末尾1）
  076                  : 打ち（0+先2桁 → 候補選択）
  u                    : 1手戻す（手順のみ）
  s file.kif           : 保存（cp932）
  solve 9 out.kif      : 全詰み筋探索→分岐KIF出力（python-shogi必要）
  possave file.pos     : 開始局面のみ保存（すぐstart可）
  posload file.pos     : 開始局面を読み込み（start済み）
  preview file.kif 80  : 保存済みKIFのプレビュー表示（先頭80行）
  batch [INPUT] 9      : 既存KIF群を一括処理（変化→分割 / 無変化→解探索→分割）。INPUT省略可
  help [topic]         : サブヘルプ（topic: setup / p / h / move / drop / save / solve）
  q                    : 終了
============================================================
"""

HELP_SOLVE = """\
[help solve]
solve は開始局面から「指定手数以内の強制詰み筋」を探索して、分岐KIFに出します。

使い方:
  start
  solve 9 out.kif

ポイント:
  ・手数は 1,3,5,7,9 のような奇数を想定（詰将棋）
  ・指定手数を超えたら「未解決」として打ち切ります
  ・先手側の候補手は「王手になる手」に限定して高速化します
  ・余詰があると分岐が増えます（KIFの変化として出ます）

注意:
  ・python-shogi が必要です（pip install python-shogi）
"""


HELP_SETUP = """\
[help setup]
局面作成（開始局面の作り方）

基本:
  show                 盤面確認
  p 55                 マス55に駒を置く（候補メニュー）
  p 55 v+R              直指定で置く（例: 後手の竜）
  p 55 .               駒を消す
  h b G 1              先手の持駒（例: 金1枚）
  turn b / turn w      手番（先手= b / 後手= w）
  start                この局面を開始局面として確定（ここから手順入力・solve可能）

駒トークンの例:
  P L N S G B R K       先手の歩/香/桂/銀/金/角/飛/玉
  vP vK                 後手の歩/玉
  +P +R                 成り（と/竜）※後手なら v+P のように書きます
  .                      空（消す）

よくある流れ（詰将棋）:
  p 42 vK
  p 44 P
  p 34 G
  h b G 1
  turn b
  start
"""

HELP_P = """\
[help p]
駒配置コマンド

形式:
  p 55           : メニューから選んで配置（おすすめ）
  p 55 v+R       : 直指定で配置（後手の竜）
  p 55 .         : 駒を消す

メニュー操作:
  - 番号で選択
  - 0 を選ぶと「直前に置いた駒」を再利用できます（高速配置用）
"""

HELP_H = """\
[help h]
持ち駒編集コマンド（局面作成時）

形式:
  h b P 2        : 先手(▲)の歩を2枚
  h w R 1        : 後手(△)の飛を1枚（※ツール内部では保持しますが、KIF出力時は“残駒自動計算”が主）
  h b G 0        : 0にするとその駒種を削除

注意:
  - K は持ち駒にできません
  - 枚数は 0以上の整数
"""

HELP_MOVE = """\
[help move]
手順入力（移動/成り）

形式:
  7776           : 7七→7六
  22331          : 2二→3三 成り（末尾の1が「成り」）

補足:
  - start 後に入力してください
  - “同” 表記は直前の着手先と同じマスのとき自動になります
"""

HELP_DROP = """\
[help drop]
手順入力（打ち）

形式:
  076            : 「0 + 行先2桁」→ そのマスに打てる駒候補を表示 → 番号で選択

例:
  055            : 5五に打つ（候補から選ぶ）

注意:
  - そのマスに駒がある場合は打てません
  - 二歩などは簡易チェックしています（python-shogi導入時はより厳密）
"""

HELP_SAVE = """\
[help save]
保存と読み込み/プレビュー

KIF保存:
  s out.kif

開始局面だけ保存（すぐstartできる）:
  possave mypos.pos
  posload mypos.pos   : 読み込み後は start 済み扱いになります

KIFプレビュー:
  preview out.kif
  preview out.kif 80  : 先頭80行だけ表示
"""

HELP_MAP = {
    "setup": HELP_SETUP,
    "p": HELP_P,
    "h": HELP_H,
    "move": HELP_MOVE,
    "drop": HELP_DROP,
    "save": HELP_SAVE,
    "solve": HELP_SOLVE,
}
# ----------------- Main -----------------

def main():
    pos = ShogiPosition()
    start_snapshot = None  # (board, hands_b, side_to_move)
    last_piece_token: Optional[str] = None
    end_result: Optional[str] = None  # "詰み" or "投了" (manual override)

    sente = input("先手名（Enterで先手）: ").strip() or ""
    gote  = input("後手名（Enterで後手）: ").strip() or ""
    print("\n" + HELP_MAIN)

    def show():
        print(f"\n手番: {'先手(▲)' if pos.side_to_move=='B' else '後手(△)'}")
        # show gote remaining preview (based on current board & sente hand)
        gote_auto = compute_gote_remaining(pos.board, pos.hands["B"])
        print(f"後手の持駒：{_hands_to_line(gote_auto)}")
        print(pos.board_to_piyo())
        print(f"先手の持駒：{pos.hands_to_piyo('B')}")
        print("ガイド: help solve / p 55 / h b P 2 / turn b / start / 7776 / 076 / s out.kif / solve 9 out.kif\n")

    while True:
        prompt = "▲ " if pos.side_to_move == "B" else "△ "
        s = input(prompt).strip()
        if not s:
            continue

        if s == "q":
            break

        if s.startswith("help"):
            t = s.split()
            if len(t) == 1:
                print(HELP_MAIN)
            else:
                topic = t[1].lower()
                print(HELP_MAP.get(topic, f"[help] topic '{topic}' は未対応です。使えるtopic: {', '.join(HELP_MAP.keys())}"))
            continue

        if s == "show":
            show()
            continue

        if s.startswith("turn "):
            t = s.split()
            if len(t) == 2 and t[1] in ("b","w"):
                pos.side_to_move = "B" if t[1] == "b" else "W"
                print("OK: 手番を設定しました")
            else:
                print("turn b または turn w")
            continue

        # end command (manual)
        if s.startswith("end"):
            t = s.split()
            if len(t) == 1:
                print("終局を選んでください: 1)詰み  2)投了")
                sel = input("選択: ").strip()
                if sel == "1":
                    end_result = "詰み"
                elif sel == "2":
                    end_result = "投了"
                else:
                    print("キャンセル")
                    continue
                print(f"OK: 終局={end_result}")
                continue
            if len(t) == 2:
                if t[1] in ("mate", "tsumi", "詰み"):
                    end_result = "詰み"
                    print("OK: 終局=詰み")
                elif t[1] in ("resign", "toryo", "投了"):
                    end_result = "投了"
                    print("OK: 終局=投了")
                else:
                    print("end mate / end resign")
                continue
            print("end / end mate / end resign")
            continue

        # p command
        if s.startswith("p "):
            t = s.split()
            if len(t) not in (2,3):
                print("形式: p 55 v+R  / p 55 .  / p 55(メニュー)")
                continue
            sq = t[1]
            if not re.fullmatch(r"[1-9][1-9]", sq):
                print("マスは 11〜99")
                continue
            f = int(sq[0]); r = int(sq[1])

            if len(t) == 3:
                tok = t[2].strip()
                try:
                    p = parse_piece_token(tok)
                    pos.set_piece((f,r), p)
                    if tok != ".":
                        last_piece_token = tok
                    print("OK")
                except Exception as e:
                    print(f"エラー: {e}")
                continue

            # menu
            cur = pos.board[(f,r)]
            cur_label = "・" if cur is None else (("△" if cur.color=="W" else "▲") + KIND_TO_PYO.get((cur.kind, cur.prom), PIECE_JP[cur.kind]))
            print(f"[{f}{r}] 現在: {cur_label}")

            menu = build_piece_menu(prefer_side=pos.side_to_move)
            for i, (tok, label) in enumerate(menu, start=1):
                mark = " *" if (last_piece_token is not None and tok == last_piece_token) else ""
                print(f"{i:2d}) {label}{mark}")
            if last_piece_token is not None:
                print("0) 直前の駒で置く")

            sel = input("選択: ").strip()
            try:
                if sel == "0":
                    if last_piece_token is None:
                        raise ValueError("直前の駒がありません")
                    tok = last_piece_token
                else:
                    if not sel.isdigit():
                        raise ValueError("番号で選んでください")
                    n = int(sel)
                    if not (1 <= n <= len(menu)):
                        raise ValueError("番号が範囲外です")
                    tok = menu[n-1][0]

                if tok == ".":
                    pos.set_piece((f,r), None)
                else:
                    p = parse_piece_token(tok)
                    pos.set_piece((f,r), p)
                    last_piece_token = tok
                print("OK")
            except Exception as e:
                print(f"エラー: {e}")
            continue

        # hand edit
        if s.startswith("h "):
            t = s.split()
            if len(t) != 4 or t[1] not in ("b","w"):
                print("形式: h b P 2   / h w R 1")
                continue
            color = "B" if t[1] == "b" else "W"
            kind = t[2].upper()
            if kind not in PIECE_JP or kind == "K":
                print("持ち駒は P L N S G B R のみ")
                continue
            try:
                n = int(t[3])
                if n < 0:
                    raise ValueError
                pos.set_hand(color, kind, n)
                print("OK")
            except:
                print("枚数は 0以上の整数")
            continue

        # start snapshot
        if s == "start":
            start_snapshot = (copy.deepcopy(pos.board), copy.deepcopy(pos.hands["B"]), pos.side_to_move)
            pos.clear_moves()
            end_result = None
            print("OK: この局面を開始局面として確定しました（ここから手順入力）")
            continue

        # undo
        if s == "u":
            if pos.undo():
                print("OK: 1手戻しました")
            else:
                print("戻せる手がありません")
            continue

        # solve command

        # start-position quick save/load (ready-to-start)
        if s.startswith("possave "):
            fn = s[len("possave "):].strip()
            if not fn.lower().endswith(".pos"):
                fn += ".pos"
            try:
                save_startpos_file(fn, pos.board, pos.hands["B"], pos.side_to_move, sente, gote)
                print(f"保存しました（開始局面）: {fn}")
            except Exception as e:
                print(f"保存エラー: {e}")
            continue

        if s.startswith("posload "):
            fn = s[len("posload "):].strip()
            try:
                board_map, hands_b, stm, se2, go2 = load_startpos_file(fn)
                pos.clear_all()
                pos.board = board_map
                pos.hands["B"] = hands_b
                pos.hands["W"] = {}  # not used (auto)
                pos.side_to_move = stm
                # set start snapshot immediately (so user can solve right away)
                start_snapshot = (copy.deepcopy(pos.board), copy.deepcopy(pos.hands["B"]), pos.side_to_move)
                end_result = None
                if se2: sente = se2
                if go2: gote = go2
                print(f"読み込みました（開始局面・start済み）: {fn}")
            except Exception as e:
                print(f"読み込みエラー: {e}")
            continue

        # KIF preview
        if s.startswith("preview "):
            t = s.split()
            if len(t) not in (2,3):
                print("形式: preview file.kif [lines]")
                continue
            fn = t[1]
            n = 120
            if len(t) == 3:
                try: n = int(t[2])
                except: pass
            try:
                preview_kif_file(fn, max_lines=n)
            except Exception as e:
                print(f"previewエラー: {e}")
            continue


        # Batch process existing KIFs: split variations or solve-and-split
        # Usage:
        #   batch INPUT_DIR [ply]
        #   batch somefile.kif [ply]
        # Notes:
        #   - if the input KIF contains '変化：' blocks, we split them into separate files.
        #   - if not, we solve for multiple mate lines within ply (default: infer from file, else 9).
        if s.startswith("batch"):
            t = s.split()
            # デフォルトは INPUT フォルダ一括
            target = None
            ply = None
            if len(t) == 1:
                target = str(_ensure_input_dir())
            elif len(t) == 2:
                # batch 9 のように数字だけ来たら ply とみなす
                try:
                    ply = int(t[1])
                    target = str(_ensure_input_dir())
                except:
                    target = t[1]
            else:
                # batch <dir|file.kif> [ply]
                target = t[1]
                try:
                    ply = int(t[2])
                except:
                    ply = None

            # limits use the same defaults as solve (安全弁)
            limits = SolveLimits(max_nodes=50000, max_time_sec=5.0, max_solutions=300)
            try:
                written = batch_process_path(target, default_ply=ply, limits=limits, check_only=True)
            except Exception as e:
                print(f"[batch] エラー: {e}")
                print("形式: batch [dir|file.kif|ply] [ply]")
                print("  例: batch            # INPUT 内を一括処理")
                print("      batch 9          # INPUT 内を 9 手で一括処理")
                print("      batch INPUT 9    # INPUT フォルダを 9 手で一括処理")
                print("      batch some.kif 3 # 単体KIFを 3 手で探索/分割")
                continue

            if not written:
                print("[batch] 対象が見つかりません（.kif が無いか、フォルダ/ファイルが存在しません）")
                continue
            print("[batch] 出力しました:")
            for w in written:
                print("  " + w)
            continue

        if s.startswith("solve"):
            if not HAS_PYSHOGI:
                print("python-shogi が必要です: pip install python-shogi")
                continue
            if start_snapshot is None:
                print("先に start で開始局面を確定してください（局面作成→start→solve）")
                continue

            t = s.split()
            if len(t) < 2:
                print("形式: solve 9 [out.kif] [--maxnodes N] [--maxtime SEC] [--maxsol N]")
                continue

            # parse args
            try:
                ply = int(t[1])
            except:
                print("形式: solve 9 [out.kif] ... （手数は整数）")
                continue

            out = None
            i = 2
            if i < len(t) and not t[i].startswith("--"):
                out = t[i]
                i += 1
                if out and not out.lower().endswith(".kif"):
                    out += ".kif"

            # defaults
            maxnodes = 50000
            maxtime = 5.0
            maxsol = 300

            # flags
            while i < len(t):
                if t[i] == "--maxnodes" and i + 1 < len(t):
                    maxnodes = int(t[i+1]); i += 2; continue
                if t[i] == "--maxtime" and i + 1 < len(t):
                    maxtime = float(t[i+1]); i += 2; continue
                if t[i] == "--maxsol" and i + 1 < len(t):
                    maxsol = int(t[i+1]); i += 2; continue
                print(f"[solve] 不明なオプション: {t[i]}")
                print("形式: solve 9 [out.kif] [--maxnodes N] [--maxtime SEC] [--maxsol N]")
                break
            else:
                # ok
                pass

            if ply <= 0:
                print("手数は正の整数で指定してください（例: solve 9）")
                continue

            board_map, hands_b, stm = start_snapshot
            gote_auto = compute_gote_remaining(board_map, hands_b)
            sfen = snapshot_to_sfen(board_map, hands_b, stm, gote_auto)

            b = shogi.Board(sfen)
            attacker_turn = b.turn  # side to move at start

            limits = SolveLimits(max_nodes=maxnodes, max_time_sec=maxtime, max_solutions=maxsol)
            stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}

            t0 = time.perf_counter()
            tree = solve_mate_tree(
                b, ply_left=ply, attacker_turn=attacker_turn, check_only=True, memo={}, stats=stats, limits=limits
            )
            elapsed = time.perf_counter() - t0

            if tree is None or not tree.children:
                msg = f"[solve] {ply}手以内の強制詰みは見つかりませんでした"
                if stats.get("cutoff"):
                    msg += "（制限で打ち切り）"
                print(msg)
                print(f"[solve] 探索: {elapsed:.3f}s / nodes={stats.get('nodes',0)}")
                continue

            # prune for output safety
            tree_out = prune_tree_to_max_leaves(tree, maxsol)
            sol = count_solutions(tree_out)

            cutoff_note = "（制限で打ち切り・部分解の可能性）" if stats.get("cutoff") else ""
            print(f"[solve] 解答筋（葉の数）: {sol} {cutoff_note}")
            print(f"[solve] 探索: {elapsed:.3f}s / nodes={stats.get('nodes',0)}")

            if out:
                # Write each solution line to separate KIF files under OUTPUT/
                board0 = shogi.Board(sfen)
                paths = enumerate_solution_paths(tree_out)
                stem = Path(out).stem
                written: List[Path] = []
                if not paths:
                    print("[solve] 解答筋がありません")
                else:
                    # First line: stem.kif, others: stem_002.kif...
                    seen_kif: Dict[str,str] = {}
                    p0 = generate_kif_single_line(board0, hands_b, gote_auto, sente, gote, paths[0], f"{stem}.kif", seen_kif)
                    if p0:
                        written.append(str(p0))
                    for i_line, mvlist in enumerate(paths[1:], start=2):
                        pw = generate_kif_single_line(board0, hands_b, gote_auto, sente, gote, mvlist, f"{stem}_{i_line:03d}.kif", seen_kif)
                        if pw:
                            written.append(str(pw))
                    if len(written) == 1:
                        print(f"[solve] KIFを OUTPUT に保存しました: {stem}.kif")
                    else:
                        print(f"[solve] KIFを {len(written)} 本、OUTPUT に保存しました: {stem}.kif + {stem}_002.kif ...")

            else:
                print("[solve] 出力ファイル名を付けると、各解答筋を別KIFでOUTPUTへ保存します。例: solve 9 solve.kif")
            continue

        # save command (manual sequence)
        if s.startswith("s "):
            fn = s[2:].strip()
            if not fn.lower().endswith(".kif"):
                fn += ".kif"
            if start_snapshot is None:
                print("先に start で開始局面を確定してください（局面作成→start→手順入力）")
                continue

            board0_map, hands0_b, stm0 = start_snapshot
            gote_auto = compute_gote_remaining(board0_map, hands0_b)

            # Try to auto-detect mate with python-shogi if available and end_result not set
            auto_end = end_result
            if auto_end is None and HAS_PYSHOGI:
                try:
                    sfen = snapshot_to_sfen(board0_map, hands0_b, stm0, gote_auto)
                    b = shogi.Board(sfen)
                    for mv in pos.moves:
                        # Convert minimal Move -> USI move string is non-trivial without strict mapping;
                        # here we simply skip auto-check in manual save path to avoid mismatches.
                        # (Solve path is the main auto path.)
                        pass
                except:
                    pass

            # Build KIF
            out: List[str] = []
            out.append("# ---- Kifu for Mac V0.53 夢の詰将棋メーカー by CUI ----")
            out.append(f"終了日時：{now_yyyy_mm_dd_hhmmss()}")
            out.append("手合割：平手")
            out.append("後手の持駒：" + _hands_to_line(gote_auto))
            out.append(_board_map_to_piyo(board0_map))
            out.append("先手の持駒：" + _hands_to_line(hands0_b))
            out.append(f"先手：{sente}")
            out.append(f"後手：{gote}")
            out.append("手数----指手---------消費時間--")

            # We continue to output the user's entered lines using our lightweight formatter (no strict legality).
            sec_per_move = 3
            total_sec = 0
            prev_to = None
            tmp_board = copy.deepcopy(board0_map)
            tmp_hands = {"B": copy.deepcopy(hands0_b), "W": copy.deepcopy(gote_auto)}
            tmp_side = stm0

            for i, mv in enumerate(pos.moves, start=1):
                total_sec += sec_per_move
                line, prev_to = _kif_line_for_minimal_move(i, tmp_board, mv, prev_to, sec_per_move, total_sec)
                out.append(line)
                # apply minimally to keep "同" consistent
                _apply_minimal_to_tmp(tmp_board, tmp_hands, tmp_side, mv)
                tmp_side = "W" if tmp_side == "B" else "B"

            # end line
            endtxt = auto_end or end_result
            if endtxt is None:
                endtxt = "詰み"  # for tsume workflow, default to mate if user forgets (can be overridden with end resign)
            out.append(f"{len(pos.moves)+1:4d} {endtxt:<12} ( 0:{sec_per_move:02d}/00:00:{total_sec:02d})")

            text = "\n".join(out) + "\n"
            outp = _resolve_kif_path(fn)
            outp.write_bytes(text.encode("cp932", errors="replace"))
            print(f"保存しました: {outp}")
            continue

        # numeric move input
        try:
            if start_snapshot is None:
                print("先に局面を作って start してください（help solve を参照）")
                continue
            mode, _, frm, to, promote = pos.parse_numeric(s)

            if mode == "drop_pick":
                cands = pos.drop_candidates(to)
                if not cands:
                    raise ValueError("そのマスに打てる駒がありません（駒あり/持ち駒なし/二歩など）")
                print(f"打ち先：{sq_to_kif(*to)}")
                for i, k in enumerate(cands, start=1):
                    print(f" {i}) {PIECE_JP[k]}")
                sel = input("選択: ").strip()
                if not sel.isdigit() or not (1 <= int(sel) <= len(cands)):
                    raise ValueError("選択が不正です")
                kind = cands[int(sel)-1]
                mv = pos.apply_move_minimal(kind, None, to, False, True)
                idx = len(pos.moves)
                print(f"{idx:4d} {sq_to_kif(*mv.to_sq)}{PIECE_JP[mv.kind]}打")
                continue

            if mode == "move":
                p = pos.board.get(frm)
                if p is None or p.color != pos.side_to_move:
                    raise ValueError("移動元に手番の駒がありません")
                mv = pos.apply_move_minimal(p.kind, frm, to, promote, False)
                idx = len(pos.moves)
                suffix = "成" if promote else ""
                print(f"{idx:4d} {sq_to_kif(*mv.to_sq)}{PIECE_JP[mv.kind]}{suffix}{sq_to_paren(*mv.from_sq)}")
                continue

        except Exception as e:
            print(f"入力エラー: {e}")

def _board_map_to_piyo(board_map: Dict[Tuple[int,int], Optional[Piece]]) -> str:
    lines = []
    lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
    lines.append("+---------------------------+")
    for r in range(1,10):
        row = []
        for f in range(9,0,-1):
            p = board_map[(f,r)]
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
        # remove from hand if present
        if mv.kind in hands[side]:
            hands[side][mv.kind] -= 1
            if hands[side][mv.kind] <= 0:
                del hands[side][mv.kind]
        board_map[mv.to_sq] = Piece(side, mv.kind, False)
    else:
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

if __name__ == "__main__":
    main()
