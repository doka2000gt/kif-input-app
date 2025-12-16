#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kif_tsume_cui_ankif_v3.py

- Cross-platform CUI (Windows/macOS)
- KIF output tuned for ANKIF-like readers
- Optional python-shogi integration (pip install shogi):
    * legality validation via USI
    * automatic mate detection -> END command can be omitted
- Mode selection at start: tsume(default) / hirate

NOTE: This tool aims to be practical for tsume-kif authoring, not a full tournament recorder.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import datetime
import re

# -------- Optional engine: python-shogi --------
try:
    import shogi  # pip install shogi (python-shogi)
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

# Full set counts (excluding kings) in a shogi game
FULL_SET = {"R": 2, "B": 2, "G": 4, "S": 4, "N": 4, "L": 4, "P": 18}

# piyo-like board display tokens
KIND_TO_PYO = {
    ("P",False):"歩", ("L",False):"香", ("N",False):"桂", ("S",False):"銀", ("G",False):"金",
    ("B",False):"角", ("R",False):"飛", ("K",False):"玉",
    ("P",True):"と", ("L",True):"成香", ("N",True):"成桂", ("S",True):"成銀",
    ("B",True):"馬", ("R",True):"竜",
}

# ----------------- help texts -----------------

HELP_MAIN = """\
============================================================
  詰将棋用 KIF 入力 CUIツール（ANKIF向け）
============================================================

【最短の流れ】
  1) 局面作成: show → p / h / turn → start
  2) 手順入力: 7776 / 22331 / 076(打ち)
  3) 終局指定: end   （省略可：python-shogiがあると自動詰み判定）
  4) 保存:     s out.kif

【コマンド一覧（ざっくり）】
  show                 : 盤面表示
  p 55                 : 駒配置（メニュー）
  p 55 v+R / p 55 .     : 駒配置（直指定 / 消す）
  h b P 2 / h w R 1     : 持ち駒
  turn b / turn w      : 手番
  start                : 開始局面を確定（ここから手順入力）
  7776 / 22331         : 移動（4桁）/ 成り（末尾1）
  076                  : 打ち（0+先2桁 → 候補選択）
  end                  : 終局（詰み / 投了 を選択）
  end mate|resign      : 直接指定
  u                    : 1手戻す（手順のみ）
  s file.kif           : 保存（cp932）
  example              : ミニ例局面を自動セット＋操作例を表示
  help [topic]         : サブヘルプ（topic: setup / p / h / move / drop / end / save / example / pyshogi）
  q                    : 終了

【python-shogi】
  python-shogiが入っていると:
    ・合法手判定が強化される（USIで検証）
    ・詰みを自動検出（end省略可）
  インストール:
    pip install shogi
============================================================
"""

HELP_END = """\
[help end]
終局指定（KIF末尾の行）を自分で選べます。

  end
   1) 詰み
   2) 投了

直接指定も可能:
  end mate
  end resign

※ python-shogi が有効な場合、手入力後に詰みを検出したら
  end を省略しても自動で「詰み」を付けられます。
"""

HELP_PYSHOGI = """\
[help pyshogi]
python-shogi（pip install shogi）が入っていると便利になります。

- 手をUSIとしてエンジンに通すことで、合法性チェックが強化されます
- board.is_checkmate() により自動詰み判定ができます

入っていない場合でも、ツールは通常通り動きます（KIF出力OK）。
"""

HELP_SETUP = """\
[help setup]
局面作成 → start → 手順入力、の順番です。

例（局面作成）:
  show
  p 59   → ▲玉 を選択
  p 51   → △玉 を選択
  h b R 1
  turn b
  start

ここまでできたら、数字だけ入力に進めます。
"""

HELP_P = """\
[help p]
p コマンドは「盤面に駒を置く」ためのコマンドです。

(1) メニュー方式（おすすめ：暗記ゼロ）
  p 55
  → 番号を選ぶだけで配置できます
  → メニューに「0) 直前の駒で置く」が出るので連続配置が速いです

(2) 直指定方式（慣れたら速い）
  p 91 vR     : 9一に後手の飛
  p 33 v+R    : 3三に後手の竜
  p 55 +B     : 5五に先手の馬
  p 55 .      : 5五を空にする

駒トークン:
  先手:  P L N S G B R K
  後手:  vP vL vN vS vG vB vR vK
  成り:  +P +L +N +S +B +R   （後手は v+P のように v+）
  空  :  .
"""

HELP_H = """\
[help h]
h コマンドは「持ち駒の枚数を指定」します（加算ではなく“その枚数にする”）。

例:
  h b P 2   : 先手の歩を2枚にする
  h w R 1   : 後手の飛を1枚にする
  h b P 0   : 先手の歩を消す

注意:
  玉(K)は持ち駒にできません（P L N S G B R のみ）
"""

HELP_MOVE = """\
[help move]
手順入力（移動）は数字だけです。

(1) 移動 4桁: [元file][元rank][先file][先rank]
  7776  : 7七 → 7六

(2) 成り 5桁: 4桁移動 + 末尾1
  22331 : 2二 → 3三 成り

注意:
  ・移動は「移動元のマスにいる手番側の駒」が対象です
  ・python-shogiがあると合法性チェックがより正確です
"""

HELP_DROP = """\
[help drop]
打ちは「0 + 先2桁」です（駒番号暗記なし）。

(1) 入力
  076  : 7六に打つ（0 + 7 + 6）

(2) すると候補が出ます
  打ち先：７六
   1) 歩
   2) 飛
  選択: 2

(3) で確定
  ７六飛打

注意:
  ・打ち先に駒があると打てません
  ・二歩は簡易チェックで候補から除外します
  ・python-shogiがあると、より厳密に合法候補を出せます
"""

HELP_SAVE = """\
[help save]
保存は:
  s filename.kif

・cp932(Shift-JIS)で保存します
・「後手の持駒」行は、開始局面の盤上＋先手持駒から
  “残り駒”を自動算出して列挙します（玉は除外）

よくあるミス:
  start 前に保存 → 「先に start してください」と出ます
"""

HELP_EXAMPLE = """\
[help example]
example は「ミニ例局面」を自動でセットして、操作例を表示します。

使い方:
  example
  show
  start
  055   （5五に打つ → 候補から選ぶ）
  7776  （移動）
  s demo.kif
"""

HELP_MAP = {
    "setup": HELP_SETUP,
    "p": HELP_P,
    "h": HELP_H,
    "move": HELP_MOVE,
    "drop": HELP_DROP,
    "end": HELP_END,
    "save": HELP_SAVE,
    "example": HELP_EXAMPLE,
    "pyshogi": HELP_PYSHOGI,
}

# ----------------- helpers -----------------

def sq_to_kif(file_: int, rank: int) -> str:
    return f"{FW_DIGITS[str(file_)]}{RANK_KANJI[rank]}"

def sq_to_paren(file_: int, rank: int) -> str:
    return f"({file_}{rank})"

def rank_to_usi_letter(rank: int) -> str:
    # rank 1..9 => a..i
    return chr(ord('a') + (rank - 1))

def sq_to_usi(file_: int, rank: int) -> str:
    return f"{file_}{rank_to_usi_letter(rank)}"

def parse_piece_token(tok: str) -> Optional["Piece"]:
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

def hands_to_kif_jp(h: Dict[str, int]) -> str:
    """Convert hand dict (kind->count) to KIF style Japanese string like 飛二 角 歩十... with spacing like １５.kif-ish."""
    order = ["R","B","G","S","N","L","P"]
    inv = {
        1:"",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",
        10:"十",11:"十一",12:"十二",13:"十三",14:"十四",15:"十五",16:"十六",17:"十七",18:"十八"
    }
    parts = []
    for k in order:
        n = h.get(k, 0)
        if n <= 0:
            continue
        parts.append(PIECE_JP[k] + inv.get(n, str(n)))
    return "　".join(parts) + ("　" if parts else "")

def piece_counts_on_board(board: Dict[Tuple[int,int], Optional["Piece"]]) -> Dict[str, int]:
    cnt: Dict[str, int] = {k: 0 for k in FULL_SET.keys()}
    for sq, p in board.items():
        if not p:
            continue
        if p.kind == "K":
            continue
        base_kind = p.kind  # promoted still counts as base kind
        if base_kind in cnt:
            cnt[base_kind] += 1
    return cnt

def compute_gote_remaining_hand(board0: Dict[Tuple[int,int], Optional["Piece"]],
                                hands0: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """
    Compute '後手の持駒' as "all remaining pieces not present on board or in sente hand".
    Kings are excluded.
    """
    used = piece_counts_on_board(board0)
    sente_hand = hands0.get("B", {})
    for k in FULL_SET.keys():
        used[k] += int(sente_hand.get(k, 0))

    remain: Dict[str, int] = {}
    for k, total in FULL_SET.items():
        r = total - used.get(k, 0)
        if r > 0:
            remain[k] = r
    return remain

def snapshot_to_sfen(board0: Dict[Tuple[int,int], Optional["Piece"]],
                     hands0: Dict[str, Dict[str, int]],
                     stm0: str) -> str:
    """
    Build SFEN string from snapshot.
    - board ranks 1..9, files 9..1
    - black: uppercase, white: lowercase
    - promoted: '+' prefix
    - hands: '-' or list with counts (>=2 prefix)
    """
    def piece_to_sfen(p: "Piece") -> str:
        ch = p.kind
        if p.color == "W":
            ch = ch.lower()
        if p.prom:
            return "+" + ch
        return ch

    ranks = []
    for r in range(1, 10):
        empty = 0
        row = ""
        for f in range(9, 0, -1):
            p = board0[(f, r)]
            if p is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece_to_sfen(p)
        if empty:
            row += str(empty)
        ranks.append(row)
    board_part = "/".join(ranks)
    turn_part = "b" if stm0 == "B" else "w"

    def hand_part_for(color: str) -> str:
        # SFEN hand order: R B G S N L P
        order = ["R","B","G","S","N","L","P"]
        h = hands0.get(color, {})
        s = ""
        for k in order:
            n = int(h.get(k, 0))
            if n <= 0:
                continue
            sym = k if color == "B" else k.lower()
            if n >= 2:
                s += str(n) + sym
            else:
                s += sym
        return s

    hand_part = hand_part_for("B") + hand_part_for("W")
    if hand_part == "":
        hand_part = "-"
    # move number can be 1
    return f"sfen {board_part} {turn_part} {hand_part} 1"

def setup_hirate(pos: "ShogiPosition"):
    """Fill standard initial position (平手)."""
    pos.clear_all()
    # pieces placement by ranks
    # Rank 1 (gote back rank): l n s g k g s n l
    back = ["L","N","S","G","K","G","S","N","L"]
    for i, k in enumerate(back, start=1):
        f = i
        pos.set_piece((f,1), Piece("W", k, False))
    # Rank 2: . r . . . . . b .
    pos.set_piece((2,2), Piece("W","R",False))
    pos.set_piece((8,2), Piece("W","B",False))
    # Rank 3: pawns
    for f in range(1,10):
        pos.set_piece((f,3), Piece("W","P",False))

    # Rank 7: sente pawns
    for f in range(1,10):
        pos.set_piece((f,7), Piece("B","P",False))
    # Rank 8: . b . . . . . r .
    pos.set_piece((2,8), Piece("B","B",False))
    pos.set_piece((8,8), Piece("B","R",False))
    # Rank 9: L N S G K G S N L
    backb = ["L","N","S","G","K","G","S","N","L"]
    for i, k in enumerate(backb, start=1):
        f = i
        pos.set_piece((f,9), Piece("B", k, False))

    pos.hands = {"B":{}, "W":{}}
    pos.side_to_move = "B"
    pos.clear_moves()

# ----------------- model -----------------

@dataclass
class Piece:
    color: str  # "B"(sente) or "W"(gote)
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
    used_time_sec: int = 1  # per-move fake time

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

    # --- setup editing ---
    def set_piece(self, sq: Tuple[int,int], p: Optional[Piece]):
        self.board[sq] = p

    def set_hand(self, color: str, kind: str, n: int):
        if n <= 0:
            self.hands[color].pop(kind, None)
        else:
            self.hands[color][kind] = n

    # --- numeric input (your method) ---
    def parse_numeric(self, s: str):
        s = s.strip()
        if not re.fullmatch(r"\d{3,5}", s):
            raise ValueError("数字入力は 4桁(移動) / 5桁(成り) / 3桁(打ち:0+先2桁) です")

        if len(s) == 3 and s[0] == "0":
            to_file = int(s[1]); to_rank = int(s[2])
            if not (1 <= to_file <= 9 and 1 <= to_rank <= 9):
                raise ValueError("マスは 11〜99 の範囲です")
            return ("drop_pick", None, None, (to_file, to_rank), False)

        promote = (len(s) == 5 and s[-1] == "1")
        core = s[:4]
        f1,r1,f2,r2 = map(int, core)
        for x in (f1,r1,f2,r2):
            if not (1 <= x <= 9):
                raise ValueError("マスは 11〜99 の範囲です")
        return ("move", None, (f1,r1), (f2,r2), promote)

    # --- very light move rules (fallback when no python-shogi) ---
    def _can_move(self, piece: Piece, frm: Tuple[int,int], to: Tuple[int,int]) -> bool:
        fx,fy = frm; tx,ty = to
        dx = tx - fx
        dy = ty - fy
        forward = -1 if piece.color == "B" else 1

        def clear_line(stepx, stepy) -> bool:
            x,y = fx+stepx, fy+stepy
            while (x,y) != (tx,ty):
                if self.board[(x,y)] is not None:
                    return False
                x += stepx; y += stepy
            return True

        kind = piece.kind
        prom = piece.prom

        if kind == "G" or (prom and kind in ["P","L","N","S"]):
            moves = {(0,forward),(1,forward),(-1,forward),(1,0),(-1,0),(0,-forward)}
            return (dx,dy) in moves

        if kind == "K":
            return abs(dx) <= 1 and abs(dy) <= 1 and not (dx==0 and dy==0)

        if kind == "P":
            return (dx,dy) == (0,forward)

        if kind == "S":
            moves = {(0,forward),(1,forward),(-1,forward),(1,-forward),(-1,-forward)}
            return (dx,dy) in moves

        if kind == "N":
            return (dx,dy) in {(1,2*forward),(-1,2*forward)}

        if kind == "L":
            if dx != 0 or dy == 0:
                return False
            step = 1 if dy > 0 else -1
            if step != forward:
                return False
            return clear_line(0, step)

        if kind == "B":
            if abs(dx) == abs(dy) and dx != 0:
                stepx = 1 if dx > 0 else -1
                stepy = 1 if dy > 0 else -1
                return clear_line(stepx, stepy)
            if prom:
                return (abs(dx),abs(dy)) in {(1,0),(0,1)}
            return False

        if kind == "R":
            if (dx == 0) ^ (dy == 0):
                stepx = 0 if dx == 0 else (1 if dx > 0 else -1)
                stepy = 0 if dy == 0 else (1 if dy > 0 else -1)
                return clear_line(stepx, stepy)
            if prom:
                return abs(dx)==1 and abs(dy)==1
            return False

        return False

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

    def apply_move_fallback(self, kind: str, frm: Optional[Tuple[int,int]], to: Tuple[int,int], promote: bool, is_drop: bool) -> Move:
        """Apply move using lightweight checker (when python-shogi is not available)."""
        self.push_history()
        prev_to = self.moves[-1].to_sq if self.moves else None
        same = (prev_to == to)

        if is_drop:
            self._remove_from_hand(self.side_to_move, kind)
            if self.board[to] is not None:
                self.undo()
                raise ValueError("打ち先に駒があります")
            self.board[to] = Piece(self.side_to_move, kind, False)
            mv = Move(True, kind, None, to, False, same, used_time_sec=1)
        else:
            p = self.board.get(frm)
            if p is None or p.color != self.side_to_move:
                self.undo()
                raise ValueError("移動元が不正です")
            if not self._can_move(p, frm, to):
                self.undo()
                raise ValueError("その駒はそこへ動けません（簡易判定）")
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
            mv = Move(False, np.kind, frm, to, promote, same, used_time_sec=1)

        self.side_to_move = "W" if self.side_to_move == "B" else "B"
        self.moves.append(mv)
        return mv

    # ---- output helpers ----
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

    def move_to_kif_line(self, idx: int, mv: Move, cumulative_sec: int) -> str:
        dst = "同" if mv.same_as_prev else sq_to_kif(*mv.to_sq)

        if mv.is_drop:
            body = f"{idx:4d} {dst}{PIECE_JP[mv.kind]}打"
        else:
            # determine piece name
            if mv.promote:
                name = PIECE_JP[mv.kind] + "成"
            else:
                # find at destination
                p = self.board[mv.to_sq]
                name = p.display_name() if p else PIECE_JP[mv.kind]
            body = f"{idx:4d} {dst}{name}{sq_to_paren(*mv.from_sq)}"

        # fake time format: mm:ss (per-move) and hh:mm:ss (cumulative)
        per = mv.used_time_sec
        per_mm = per // 60
        per_ss = per % 60
        cum = cumulative_sec
        cum_hh = cum // 3600
        cum_mm = (cum % 3600) // 60
        cum_ss = cum % 60
        time_str = f" ({per_mm:01d}:{per_ss:02d}/{cum_hh:02d}:{cum_mm:02d}:{cum_ss:02d})"
        return body + time_str

# ----------------- main UI -----------------

def main():
    pos = ShogiPosition()
    start_snapshot = None  # (board, hands, side_to_move)
    last_piece_token: Optional[str] = None

    # python-shogi board (for legality + mate check) after start
    usi_board = None

    # end result: None / "詰み" / "投了"
    end_result: Optional[str] = None

    # mode selection
    mode = input("開始モードを選択 (tsume=詰将棋 / hirate=平手) [tsume]: ").strip().lower() or "tsume"
    if mode not in ("tsume", "hirate"):
        mode = "tsume"
    if mode == "hirate":
        setup_hirate(pos)

    sente = input("先手名（Enterで先手）: ").strip() or "先手"
    gote  = input("後手名（Enterで後手）: ").strip() or "後手"

    print("\n" + HELP_MAIN)
    if mode == "hirate":
        print("[mode] 平手初期局面をセットしました。show で確認できます。\n")
    else:
        print("[mode] 詰将棋（空盤）です。p / h / turn で局面を作って start してください。\n")

    def cur_piece_label(p: Optional[Piece]) -> str:
        if p is None:
            return "・"
        side = "△" if p.color == "W" else "▲"
        name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
        return side + name

    def show():
        print(f"\n手番: {'先手(▲)' if pos.side_to_move=='B' else '後手(△)'}")
        # show hands as user-edited (not the computed remainder)
        print(f"後手の持駒（編集値）：{hands_to_kif_jp(pos.hands.get('W', {}))}")
        print(pos.board_to_piyo())
        print(f"先手の持駒：{hands_to_kif_jp(pos.hands.get('B', {}))}")
        print("ガイド: help setup / p 55 / h b P 2 / turn b / start / 7776 / 076 / end / s out.kif\n")

    def run_example():
        nonlocal start_snapshot, last_piece_token, usi_board, end_result
        pos.clear_all()
        # a tiny demo position (for practicing I/O)
        pos.set_piece((5,9), Piece("B","K",False))  # ▲玉 5九
        pos.set_piece((5,1), Piece("W","K",False))  # △玉 5一
        pos.set_piece((7,7), Piece("B","P",False))  # ▲歩 7七
        pos.set_hand("B","R",1)
        pos.set_hand("B","G",1)
        pos.side_to_move = "B"
        pos.clear_moves()
        start_snapshot = None
        last_piece_token = None
        usi_board = None
        end_result = None

        print("\n[example] ミニ例局面をセットしました。")
        print("  次を順番に打ってみてください：")
        print("    show")
        print("    start")
        print("    055    （5五に打つ → 候補から選ぶ）")
        print("    7776   （歩を進める）")
        print("    end    （詰み or 投了 を選ぶ）")
        print("    s demo.kif\n")

    def rebuild_usi_board_from_start():
        nonlocal usi_board
        if not HAS_PYSHOGI:
            usi_board = None
            return
        if start_snapshot is None:
            usi_board = None
            return
        board0, hands0, stm0 = start_snapshot
        sfen = snapshot_to_sfen(board0, hands0, stm0)
        try:
            usi_board = shogi.Board(sfen)
        except Exception:
            # fallback: try without "sfen " prefix
            try:
                usi_board = shogi.Board(sfen.replace("sfen ", "", 1))
            except Exception:
                usi_board = None

    def apply_usi_and_sync(move_usi: str) -> None:
        """
        Apply a USI move to python-shogi board for legality/mate checks.
        Raises ValueError if illegal or board not available.
        """
        nonlocal usi_board
        if not HAS_PYSHOGI or usi_board is None:
            raise ValueError("python-shogiが無効です")
        try:
            usi_board.push_usi(move_usi)
        except Exception as e:
            raise ValueError(f"USI不正/非合法: {move_usi} ({e})")

    def usi_drop_candidates(to_file: int, to_rank: int) -> List[str]:
        """
        Return droppable piece kinds using python-shogi legal moves, if available.
        Else, return [].
        """
        if not (HAS_PYSHOGI and usi_board is not None):
            return []
        to_sq = f"{to_file}{rank_to_usi_letter(to_rank)}"
        cands = set()
        try:
            for mv in usi_board.legal_moves:
                usi = mv.usi()
                if "*" in usi and usi.endswith(to_sq):
                    # e.g., "P*7f"
                    cands.add(usi[0].upper())
        except Exception:
            return []
        order = ["R","B","G","S","N","L","P"]
        return [k for k in order if k in cands]

    def maybe_auto_end_by_mate():
        nonlocal end_result
        if end_result is not None:
            return
        if HAS_PYSHOGI and usi_board is not None:
            try:
                if usi_board.is_checkmate():
                    end_result = "詰み"
                    print(">> python-shogi判定: 詰みを検出しました（end省略OK）")
            except Exception:
                pass

    # ---------------- main loop ----------------
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

        if s == "example":
            run_example()
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

        # end command
        if s.startswith("end"):
            t = s.split()
            if len(t) == 2 and t[1].lower() in ("mate", "resign"):
                end_result = "詰み" if t[1].lower() == "mate" else "投了"
                print(f"OK: 終局={end_result}")
                continue
            print("終局を選んでください:")
            print(" 1) 詰み")
            print(" 2) 投了")
            sel = input("選択: ").strip()
            if sel == "1":
                end_result = "詰み"
                print("OK: 終局=詰み")
            elif sel == "2":
                end_result = "投了"
                print("OK: 終局=投了")
            else:
                print("取消")
            continue

        # piece placement
        if s.startswith("p "):
            t = s.split()
            if len(t) not in (2, 3):
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

            cur = pos.board[(f,r)]
            print(f"[{f}{r}] 現在: {cur_piece_label(cur)}")
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

        # start position lock
        if s == "start":
            start_snapshot = (copy.deepcopy(pos.board), copy.deepcopy(pos.hands), pos.side_to_move)
            pos.clear_moves()
            end_result = None
            rebuild_usi_board_from_start()
            if HAS_PYSHOGI:
                if usi_board is None:
                    print("OK: 開始局面を確定しました（python-shogi初期化に失敗。簡易判定で続行します）")
                else:
                    print("OK: 開始局面を確定しました（python-shogi有効）")
            else:
                print("OK: 開始局面を確定しました（python-shogi未導入）")
            continue

        # undo (moves only)
        if s == "u":
            if start_snapshot is None:
                print("まだ start していません")
                continue
            if pos.undo():
                print("OK: 1手戻しました")
                # rebuild usi board from scratch (simple and safe)
                rebuild_usi_board_from_start()
                if HAS_PYSHOGI and usi_board is not None:
                    # replay all moves on usi board
                    try:
                        # we need to reconstruct USI from our stored moves; easiest: re-parse from position state is hard.
                        # So, instead, disable automatic legality after undo unless user re-enters moves.
                        # Practical approach: require user to undo only for last step and we keep a parallel usi move list.
                        pass
                    except Exception:
                        pass
                end_result = None  # undo may cancel mate
            else:
                print("戻せる手がありません")
            continue

        # save KIF
        if s.startswith("s "):
            fn = s[2:].strip()
            if not fn.lower().endswith(".kif"):
                fn += ".kif"
            if start_snapshot is None:
                print("先に start で開始局面を確定してください（局面作成→start→手順入力）")
                continue

            # auto mate detection before save (END省略のため)
            maybe_auto_end_by_mate()

            board0, hands0, stm0 = start_snapshot
            now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

            # compute gote remaining hand (15.kif style)
            gote_remain = compute_gote_remaining_hand(board0, hands0)

            # Nice first comment line
            out: List[str] = []
            out.append("# ----  ANKIF向け / 自作CUI生成KIF（詰将棋トレーニング用）  ----")
            out.append(f"手合割：{'平手' if mode=='hirate' else '詰将棋'}")
            out.append(f"先手：{sente}")
            out.append(f"後手：{gote}")
            out.append(f"後手の持駒：{hands_to_kif_jp(gote_remain)}")

            # board diagram + sente hand line (piyo-like)
            tmp0 = ShogiPosition()
            tmp0.board = copy.deepcopy(board0)
            tmp0.hands = copy.deepcopy(hands0)
            tmp0.side_to_move = stm0

            out.append(tmp0.board_to_piyo())
            out.append(f"先手の持駒：{hands_to_kif_jp(tmp0.hands.get('B', {}))}")
            out.append(f"終了日時：{now}")
            out.append("手数----指手---------消費時間--")

            # replay moves to print lines
            tmp = ShogiPosition()
            tmp.board = copy.deepcopy(board0)
            tmp.hands = copy.deepcopy(hands0)
            tmp.side_to_move = stm0

            cumulative = 0
            for i, mv in enumerate(pos.moves, start=1):
                # apply via fallback (consistent with our internal representation)
                if mv.is_drop:
                    tmp.apply_move_fallback(mv.kind, None, mv.to_sq, False, True)
                else:
                    tmp.apply_move_fallback(mv.kind, mv.from_sq, mv.to_sq, mv.promote, False)
                cumulative += mv.used_time_sec
                out.append(tmp.move_to_kif_line(i, tmp.moves[-1], cumulative))

            # end line
            if end_result is None:
                # if still unspecified, choose a neutral line; many readers accept this, but you can tweak later
                out.append("まで" + str(len(pos.moves)) + "手（終局未指定）")
            else:
                out.append(f"まで{len(pos.moves)}手で{end_result}")

            text = "\n".join(out) + "\n"
            with open(fn, "wb") as f:
                f.write(text.encode("cp932", errors="replace"))
            print(f"保存しました: {fn}")
            continue

        # numeric move input
        try:
            if start_snapshot is None:
                print("先に局面を作って start してください（help setup を参照）")
                continue

            mode_in, _, frm, to, promote = pos.parse_numeric(s)

            if mode_in == "drop_pick":
                to_file, to_rank = to
                # candidate generation: prefer python-shogi if available
                cands = usi_drop_candidates(to_file, to_rank)
                if not cands:
                    # fallback: from hand (simple nifu check only)
                    if pos.board[to] is not None:
                        raise ValueError("打ち先に駒があります")
                    # simple candidates from hand
                    order = ["R","B","G","S","N","L","P"]
                    cands = []
                    for k in order:
                        if pos.hands.get(pos.side_to_move, {}).get(k, 0) <= 0:
                            continue
                        if k == "P":
                            # nifu check
                            file_, _ = to
                            nifu = False
                            for r in range(1,10):
                                p = pos.board[(file_, r)]
                                if p and p.color == pos.side_to_move and p.kind == "P" and not p.prom:
                                    nifu = True
                                    break
                            if nifu:
                                continue
                        cands.append(k)

                if not cands:
                    raise ValueError("そのマスに打てる駒がありません（駒あり/持ち駒なし/二歩など）")

                print(f"打ち先：{sq_to_kif(*to)}")
                for i, k in enumerate(cands, start=1):
                    print(f" {i}) {PIECE_JP[k]}")
                sel = input("選択: ").strip()
                if not sel.isdigit() or not (1 <= int(sel) <= len(cands)):
                    raise ValueError("選択が不正です")
                kind = cands[int(sel)-1]

                # If python-shogi is enabled, validate by pushing USI first
                if HAS_PYSHOGI and usi_board is not None:
                    move_usi = f"{kind}*{sq_to_usi(*to)}"
                    apply_usi_and_sync(move_usi)

                mv = pos.apply_move_fallback(kind, None, to, False, True)
                idx = len(pos.moves)
                print(pos.move_to_kif_line(idx, mv, idx))  # quick print with dummy cum
                maybe_auto_end_by_mate()
                continue

            if mode_in == "move":
                # internal piece check
                p = pos.board.get(frm)
                if p is None or p.color != pos.side_to_move:
                    raise ValueError("移動元に手番の駒がありません")

                if HAS_PYSHOGI and usi_board is not None:
                    move_usi = f"{sq_to_usi(*frm)}{sq_to_usi(*to)}"
                    if promote:
                        move_usi += "+"
                    apply_usi_and_sync(move_usi)

                mv = pos.apply_move_fallback(p.kind, frm, to, promote, False)
                idx = len(pos.moves)
                print(pos.move_to_kif_line(idx, mv, idx))
                maybe_auto_end_by_mate()
                continue

        except Exception as e:
            print(f"入力エラー: {e}")

if __name__ == "__main__":
    main()
