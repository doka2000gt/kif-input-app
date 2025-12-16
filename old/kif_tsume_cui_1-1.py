#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import datetime
import re

# ----------------- constants -----------------

FW_DIGITS = {str(i): ch for i, ch in enumerate("０１２３４５６７８９")}
RANK_KANJI = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}

PIECE_JP = {"P":"歩","L":"香","N":"桂","S":"銀","G":"金","B":"角","R":"飛","K":"玉"}
PROMOTED_JP = {"P":"と","L":"成香","N":"成桂","S":"成銀","B":"馬","R":"竜"}
PROMOTABLE = set(["P","L","N","S","B","R"])

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
  詰将棋用 KIF 入力 CUIツール
============================================================

【最短の流れ】
  1) 局面作成: show → p / h / turn → start
  2) 手順入力: 7776 / 22331 / 076(打ち)
  3) 保存:     s out.kif

【コマンド一覧（ざっくり）】
  show                 : 盤面表示
  p 55                 : 駒配置（メニュー）
  p 55 v+R / p 55 .     : 駒配置（直指定 / 消す）
  h b P 2 / h w R 1     : 持ち駒
  turn b / turn w      : 手番
  start                : 開始局面を確定（ここから手順入力）
  7776 / 22331         : 移動（4桁）/ 成り（末尾1）
  076                  : 打ち（0+先2桁 → 候補選択）
  u                    : 1手戻す（手順のみ）
  s file.kif           : 保存（cp932）
  example              : ミニ例局面を自動セット＋操作例を表示
  help [topic]         : サブヘルプ（topic: setup / p / h / move / drop / save / example）
  q                    : 終了
============================================================
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
  2838  : 2八 → 3八

(2) 成り 5桁: 4桁移動 + 末尾1
  22331 : 2二 → 3三 成り

注意:
  ・移動は「移動元のマスにいる手番側の駒」が対象です
  ・このツールは詰将棋入力支援用の“軽い”移動判定です（厳密な王手回避などは未実装）
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
"""

HELP_SAVE = """\
[help save]
保存は:
  s filename.kif

・cp932(Shift-JIS)で保存します（ぴよ将棋互換寄せ）
・保存内容には「開始局面（盤面＋持ち駒）」と「指し手」が入ります

よくあるミス:
  start 前に保存 → 「先に start してください」と出ます
"""

HELP_EXAMPLE = """\
[help example]
example は「ミニ例局面」を自動でセットして、操作例を表示します。
局面を自分で作る前に、まず触ってみたい時に便利です。

使い方:
  example
  show
  start
  076   （打ち：候補から選ぶ）
  7776  （移動）
  s demo.kif
"""

HELP_MAP = {
    "setup": HELP_SETUP,
    "p": HELP_P,
    "h": HELP_H,
    "move": HELP_MOVE,
    "drop": HELP_DROP,
    "save": HELP_SAVE,
    "example": HELP_EXAMPLE,
}

# ----------------- helpers -----------------

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

    # ---- numeric input (your method) ----
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

    # ---- movement rules (lightweight; good for tsume input assistance) ----
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

        # gold or promoted P/L/N/S behaves like gold
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
            # diagonal
            if abs(dx) == abs(dy) and dx != 0:
                stepx = 1 if dx > 0 else -1
                stepy = 1 if dy > 0 else -1
                return clear_line(stepx, stepy)
            # horse additional orthogonal 1-step
            if prom:
                return (abs(dx),abs(dy)) in {(1,0),(0,1)}
            return False

        if kind == "R":
            # orthogonal
            if (dx == 0) ^ (dy == 0):
                stepx = 0 if dx == 0 else (1 if dx > 0 else -1)
                stepy = 0 if dy == 0 else (1 if dy > 0 else -1)
                return clear_line(stepx, stepy)
            # dragon additional diagonal 1-step
            if prom:
                return abs(dx)==1 and abs(dy)==1
            return False

        return False

    def drop_candidates(self, to: Tuple[int,int]) -> List[str]:
        """Return droppable piece kinds in hand (very light checks; nifu only)."""
        if self.board[to] is not None:
            return []
        kinds = sorted(self.hands[self.side_to_move].keys(), key=lambda k: "RBGSLNP".find(k) if k in "RBGSLNP" else 99)
        result: List[str] = []
        for k in kinds:
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

    def apply_move(self, kind: str, frm: Optional[Tuple[int,int]], to: Tuple[int,int], promote: bool, is_drop: bool) -> Move:
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
            mv = Move(False, np.kind, frm, to, promote, same)

        self.side_to_move = "W" if self.side_to_move == "B" else "B"
        self.moves.append(mv)
        return mv

    # ---- output helpers ----
    def hands_to_piyo(self, color: str) -> str:
        # 015.kif に寄せた表記: 「飛二 金二 ... 」(半角スペース区切り、末尾もスペース)
        order = ["R","B","G","S","N","L","P"]
        inv = {
            1:"",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",
            10:"十",11:"十一",12:"十二",13:"十三",14:"十四",15:"十五",16:"十六",17:"十七",18:"十八"
        }
        parts = []
        for k in order:
            n = self.hands[color].get(k, 0)
            if n <= 0:
                continue
            parts.append(PIECE_JP[k] + inv.get(n, str(n)))
        return (" ".join(parts) + " ") if parts else ""

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

    def move_to_kif_line(self, idx: int, mv: Move) -> str:
        dst = "同" if mv.same_as_prev else sq_to_kif(*mv.to_sq)
        if mv.is_drop:
            body = f"{idx:4d} {dst}{PIECE_JP[mv.kind]}打"
        else:
            p = self.board[mv.to_sq]
            if mv.promote:
                name = PIECE_JP[mv.kind] + "成"
            else:
                name = p.display_name() if p else PIECE_JP[mv.kind]
            body = f"{idx:4d} {dst}{name}{sq_to_paren(*mv.from_sq)}"
        return body

# ----------------- UI -----------------

def main():
    pos = ShogiPosition()
    start_snapshot = None  # (board, hands, side_to_move)
    last_piece_token: Optional[str] = None  # for quick repeat in p-menu

    sente = input("先手名（Enterで先手）: ").strip() or "先手"
    gote  = input("後手名（Enterで後手）: ").strip() or "後手"
    print("\n" + HELP_MAIN)

    def show():
        print(f"\n手番: {'先手(▲)' if pos.side_to_move=='B' else '後手(△)'}")
        print(f"後手の持駒：{pos.hands_to_piyo('W')}")
        print(pos.board_to_piyo())
        print(f"先手の持駒：{pos.hands_to_piyo('B')}")
        print("ガイド: help setup / p 55 / h b P 2 / turn b / start / 7776 / 076 / s out.kif\n")

    def cur_piece_label(p: Optional[Piece]) -> str:
        if p is None:
            return "・"
        side = "△" if p.color == "W" else "▲"
        name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
        return side + name

    def run_example():
        nonlocal start_snapshot, last_piece_token
        # Overwrite current work intentionally (example is for learning the tool)
        pos.clear_all()
        # A tiny demo position (not a rigorous tsume; just to practice input)
        # Kings
        pos.set_piece((5,9), Piece("B","K",False))  # ▲玉 5九
        pos.set_piece((5,1), Piece("W","K",False))  # △玉 5一
        # A pawn to move
        pos.set_piece((7,7), Piece("B","P",False))  # ▲歩 7七
        # Hands: sente has R and G so drop menu has options
        pos.set_hand("B","R",1)
        pos.set_hand("B","G",1)
        pos.side_to_move = "B"
        pos.clear_moves()
        start_snapshot = None
        last_piece_token = None

        print("\n[example] ミニ例局面をセットしました。")
        print("  次を順番に打ってみてください：")
        print("    show")
        print("    start")
        print("    055    （5五に打つ → 候補から「飛」か「金」を選ぶ）")
        print("    7776   （歩を進める）")
        print("    s demo.kif")
        print("  ※exampleは練習用なので、入力途中の作業は上書きされます。\n")

    while True:
        prompt = "▲ " if pos.side_to_move == "B" else "△ "
        s = input(prompt).strip()
        if not s:
            continue

        # quit
        if s == "q":
            break

        # help
        if s.startswith("help"):
            t = s.split()
            if len(t) == 1:
                print(HELP_MAIN)
            else:
                topic = t[1].lower()
                print(HELP_MAP.get(topic, f"[help] topic '{topic}' は未対応です。使えるtopic: {', '.join(HELP_MAP.keys())}"))
            continue

        # example
        if s == "example":
            run_example()
            continue

        # show
        if s == "show":
            show()
            continue

        # turn
        if s.startswith("turn "):
            t = s.split()
            if len(t) == 2 and t[1] in ("b","w"):
                pos.side_to_move = "B" if t[1] == "b" else "W"
                print("OK: 手番を設定しました")
            else:
                print("turn b または turn w")
            continue

        # piece placement
        if s.startswith("p "):
            # p 55 v+R / p 55 . / p 55 (menu)
            t = s.split()
            if len(t) not in (2, 3):
                print("形式: p 55 v+R  / p 55 .  / p 55(メニュー)")
                continue

            sq = t[1]
            if not re.fullmatch(r"[1-9][1-9]", sq):
                print("マスは 11〜99")
                continue
            f = int(sq[0]); r = int(sq[1])

            # direct token mode
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

            # menu mode
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
            # h b P 2
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
            print("OK: この局面を開始局面として確定しました（ここから手順入力）")
            continue

        # undo (moves only)
        if s == "u":
            if pos.undo():
                print("OK: 1手戻しました")
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

            board0, hands0, stm0 = start_snapshot

            out: List[str] = []
            # 015.kif 互換寄せ（ANKIFでの読み込み実績フォーマット）
            out.append("# ---- ANKIF向け 爆速KIF（kif_tsume_cui）----")
            out.append("終了日時：" + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            out.append("手合割：平手")

            tmp0 = ShogiPosition()
            tmp0.board = copy.deepcopy(board0)
            tmp0.hands = copy.deepcopy(hands0)
            tmp0.side_to_move = stm0

            out.append(f"後手の持駒：{tmp0.hands_to_piyo('W')}")
            out.append(tmp0.board_to_piyo())
            out.append(f"先手の持駒：{tmp0.hands_to_piyo('B')}")
            out.append(f"先手：{sente}")
            out.append(f"後手：{gote}")
            out.append("手数----指手---------消費時間--")

            tmp = ShogiPosition()
            tmp.board = copy.deepcopy(board0)
            tmp.hands = copy.deepcopy(hands0)
            tmp.side_to_move = stm0

            for i, mv in enumerate(pos.moves, start=1):
                if mv.is_drop:
                    tmp.apply_move(mv.kind, None, mv.to_sq, False, True)
                else:
                    tmp.apply_move(mv.kind, mv.from_sq, mv.to_sq, mv.promote, False)

                # 消費時間（架空）：1手 1秒。合計は 00:00:SS 形式。
                per_sec = 1
                total_sec = i * per_sec
                hh = total_sec // 3600
                mm = (total_sec % 3600) // 60
                ss = total_sec % 60
                total_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
                base = tmp.move_to_kif_line(i, tmp.moves[-1])
                out.append(f"{base:<18} ( 0:01/{total_str})")

            # 詰将棋用途を想定して、末尾に「詰み」を付与（015.kifの形式に合わせる）
            per_sec = 1
            total_sec = (len(pos.moves) + 1) * per_sec
            hh = total_sec // 3600
            mm = (total_sec % 3600) // 60
            ss = total_sec % 60
            total_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
            out.append(f"{len(pos.moves)+1:4d} 詰み         ( 0:01/{total_str})")
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
                mv = pos.apply_move(kind, None, to, False, True)
                idx = len(pos.moves)
                print(pos.move_to_kif_line(idx, mv))
                continue

            if mode == "move":
                p = pos.board.get(frm)
                if p is None or p.color != pos.side_to_move:
                    raise ValueError("移動元に手番の駒がありません")
                mv = pos.apply_move(p.kind, frm, to, promote, False)
                idx = len(pos.moves)
                print(pos.move_to_kif_line(idx, mv))
                continue

        except Exception as e:
            print(f"入力エラー: {e}")

if __name__ == "__main__":
    main()
