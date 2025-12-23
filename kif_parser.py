from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re

from models import Piece

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
