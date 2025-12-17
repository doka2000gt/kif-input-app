#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from constants import FW_DIGITS, RANK_KANJI, PIECE_JP
from models import Piece


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


def inv_count_kanji(n: int) -> str:
    inv = {
        1:"",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",
        10:"十",11:"十一",12:"十二",13:"十三",14:"十四",15:"十五",16:"十六",17:"十七",18:"十八"
    }
    return inv.get(n, str(n))


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


def _dedup_key_from_kif_text(kif_text: str) -> str:
    b = kif_text.encode("cp932", errors="replace")
    return hashlib.sha1(b).hexdigest()


def _write_kif_unique(
    outdir: Path,
    filename: str,
    kif_text: str,
    seen: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    key = _dedup_key_from_kif_text(kif_text)
    if seen is not None and key in seen:
        return None
    p = outdir / filename
    p.write_bytes(kif_text.encode("cp932", errors="replace"))
    if seen is not None:
        seen[key] = str(p)
    return str(p)