#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import re

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize, Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Static, Input, RichLog

from constants import PIECE_JP, KIND_TO_PYO
from position import ShogiPosition
from models import Piece
from sfen import sfen_to_snapshot, snapshot_to_sfen
from manual_kif import kif_line_for_minimal_move


# ----------------- helpers -----------------

Square = Tuple[int, int]


def _cell_str(pos: ShogiPosition, f: int, r: int) -> str:
    """盤面セルの文字列（2文字想定：' ・' / ' 歩' / 'v歩' / 'v竜' など）"""
    p = pos.board.get((f, r))
    if p is None:
        return " ・"
    name = KIND_TO_PYO.get((p.kind, p.prom), PIECE_JP[p.kind])
    return ("v" + name) if p.color == "W" else (" " + name)


def _get_hand_dict(pos: ShogiPosition, color: str) -> Dict[str, int]:
    """
    pos.hands / pos.hand のどちらでも動くように吸収
    想定: {"P":3, "R":1, ...}
    """
    if hasattr(pos, "hands") and isinstance(getattr(pos, "hands"), dict):
        return getattr(pos, "hands").get(color, {})  # type: ignore
    if hasattr(pos, "hand") and isinstance(getattr(pos, "hand"), dict):
        return getattr(pos, "hand").get(color, {})  # type: ignore
    return {}


def _format_hand(pos: ShogiPosition, color: str) -> str:
    """持駒を '飛1 角1 金2 歩3' みたいに整形（0枚は省略）"""
    order = ["R", "B", "G", "S", "N", "L", "P"]
    hand = _get_hand_dict(pos, color)

    parts = []
    for k in order:
        n = int(hand.get(k, 0) or 0)
        if n > 0:
            parts.append(f"{PIECE_JP[k]}{n}")
    return " ".join(parts) if parts else "なし"


# ----------------- Piece token parsing -----------------

def parse_piece_token(tok: str) -> Optional[Piece]:
    """
    tok examples:
      "." , "P", "+P", "vP", "v+R", "K" ...
    """
    tok = tok.strip()
    if tok in (".", "0", "none", "None"):
        return None

    color = "B"
    prom = False

    if tok.startswith("v"):
        color = "W"
        tok = tok[1:]

    if tok.startswith("+"):
        prom = True
        tok = tok[1:]

    kind = tok.upper()
    if kind not in ("P", "L", "N", "S", "G", "B", "R", "K"):
        raise ValueError(f"unknown piece token: {tok}")
    return Piece(color=color, kind=kind, prom=prom)


# ----------------- UI components -----------------

class Mode(Enum):
    NORMAL = "NORMAL"
    INPUT = "INPUT"


@dataclass
class Cursor:
    file: int = 5
    rank: int = 5


class BoardView(Static):
    def __init__(self, tui: "ShogiTui"):
        super().__init__()
        self.tui = tui

    def render(self) -> Text:
        pos = self.tui.pos
        cur = self.tui.cursor
        t = Text()

        t.append(f"▽持駒: {_format_hand(pos, 'W')}\n")
        for r in range(1, 10):
            row = []
            for f in range(9, 0, -1):
                cell = _cell_str(pos, f, r)
                if (f, r) == (cur.file, cur.rank):
                    # カーソル位置を反転表示
                    row.append(Text(cell, style="reverse"))
                else:
                    row.append(Text(cell))
            for x in row:
                t.append_text(x)
            t.append("\n")
        t.append(f"△持駒: {_format_hand(pos, 'B')}\n")
        return t


class KifViewer(ModalScreen[None]):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(self.text))

    async def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ----------------- main app -----------------

class ShogiTui(App):
    CSS = """
    Screen { layout: vertical; }
    #root { height: 1fr; layout: horizontal; }
    #left { width: 38; }
    #right { width: 1fr; }
    #cmd { height: 3; }
    #log { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit", show=False),
        Binding("i", "enter_input", "input", show=False),
        Binding("escape", "leave_input", "normal", show=False),
        Binding("h", "left", "left", show=False),
        Binding("j", "down", "down", show=False),
        Binding("k", "up", "up", show=False),
        Binding("l", "right", "right", show=False),
    ]

    mode = reactive(Mode.NORMAL)

    def __init__(self) -> None:
        super().__init__()
        self.pos = ShogiPosition()
        self.cursor = Cursor()
        self.board_view = BoardView(self)

        self.cmd_input: Optional[Input] = None
        self.log_widget: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="root"):
            with Vertical(id="left"):
                yield self.board_view
            with Vertical(id="right"):
                self.log_widget = RichLog(id="log", wrap=True)
                yield self.log_widget
                self.cmd_input = Input(placeholder="i で入力 / :q で終了", id="cmd")
                yield self.cmd_input

    def on_mount(self) -> None:
        self.mode = Mode.NORMAL
        if self.cmd_input:
            self.cmd_input.display = False
        self.board_view.refresh()
        self._set_title()
        self.log_ok("起動しました")

    def _set_title(self) -> None:
        self.title = f"将棋KIF入力TUI  [{self.mode.value}]"

    # --- logging helpers (smoke契約タグ) ---
    def _log(self, msg: str) -> None:
        if self.log_widget:
            self.log_widget.write(msg)

    def log_ok(self, msg: str) -> None:
        self._log(f"[OK] {msg}")

    def log_err(self, msg: str) -> None:
        self._log(f"[ERR] {msg}")

    def log_state(self, msg: str) -> None:
        self._log(f"[STATE] {msg}")

    def log_sfen(self, sfen: str) -> None:
        self._log(f"[SFEN] {sfen}")

    def log_kif(self, msg: str) -> None:
        self._log(f"[KIF] {msg}")

    # --- mode actions ---
    def action_enter_input(self) -> None:
        self.mode = Mode.INPUT
        self._set_title()
        if self.cmd_input:
            self.cmd_input.display = True
            self.cmd_input.focus()

    def action_leave_input(self) -> None:
        self.mode = Mode.NORMAL
        self._set_title()
        if self.cmd_input:
            self.cmd_input.value = ""
            self.cmd_input.display = False
        self.set_focus(None)

    async def action_quit(self) -> None:
        await super().action_quit()

    # --- cursor move actions ---
    def action_left(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.file = max(1, self.cursor.file - 1)
        self.board_view.refresh()

    def action_right(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.file = min(9, self.cursor.file + 1)
        self.board_view.refresh()

    def action_up(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.rank = max(1, self.cursor.rank - 1)
        self.board_view.refresh()

    def action_down(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.rank = min(9, self.cursor.rank + 1)
        self.board_view.refresh()

    # ---- KIF (A: 簡易) ----
    def _generate_kif_text(self) -> str:
        """入力作業中の確認用：簡易KIF（A）。

        最終的な「完全KIFエクスポート（B）」は、別コマンドとして拡張予定。
        """
        out: list[str] = []
        out.append("手合割：平手")
        out.append("先手：")
        out.append("後手：")
        out.append("手数----指手---------消費時間--")

        prev_to = None
        for idx, mv in enumerate(self.pos.moves, start=1):
            try:
                line = kif_line_for_minimal_move(idx, mv, prev_to, 0, 0)
            except Exception:
                line = f"{idx} {mv}"
            out.append(line)
            prev_to = getattr(mv, "to_sq", None)

        out.append(f"まで{len(self.pos.moves)}手")
        return "\n".join(out)

    # ---- Command input (クラス内メソッドとして動作させる) ----
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""

        if not line:
            return

        # 入力内容はログに残す（契約外、デバッグ用）
        self._log(f"> {line}")

        # --- numeric move input ---
        if re.fullmatch(r"\d{3,5}", line):
            try:
                tag, kind, frm, to, promote = self.pos.parse_numeric(line)

                # drop picker はここでは省略（既存設計に合わせる）
                if tag == "drop_pick":
                    cands = self.pos.drop_candidates(to)
                    if not cands:
                        raise ValueError("打てる持ち駒がありません（またはそのマスに駒があります）")
                    if len(cands) == 1:
                        self.pos.apply_move_minimal(cands[0], None, to, False, True)
                        self.log_ok(f"drop {cands[0]} -> {to}")
                        self.board_view.refresh()
                        return
                    raise ValueError("複数候補の打ち分けは未対応（smokeでは1候補前提）")

                # normal move
                p = self.pos.board.get(frm)
                if p is None:
                    raise ValueError("移動元に駒がありません")
                self.pos.apply_move_minimal(p.kind, frm, to, promote, False)
                self.log_ok(f"move {frm}->{to}{' promote' if promote else ''}")
                self.board_view.refresh()
                return

            except Exception as e:
                self.log_err(str(e))
                return

        # --- command router ---
        cmd, *rest = line.split(maxsplit=1)
        arg = rest[0] if rest else ""

        if cmd in (":q", ":quit"):
            self.mode = Mode.NORMAL
            await super().action_quit()
            return

        if cmd in ("q", "quit", "exit"):
            self.log_err("INPUT中の q は終了しません。:q を入力してください（または Ctrl+C）")
            return

        if cmd in ("help", "?"):
            self._log("[HINT] Commands: show | start | sfen | load <SFEN> | clear | undo | kif | help/? | :q")
            self._log("[HINT] Moves: 7776 (move) / 77761 (promote) / 076 (drop: 0+file+rank)")
            return

        if cmd == "show":
            self.log_state(f"手番={self.pos.side_to_move}, 手数={len(self.pos.moves)}")
            return

        if cmd == "start":
            try:
                start_sfen = "lnsgkgsnl/1r5b1/p1ppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b - 1"
                snap = sfen_to_snapshot(start_sfen)
                self.pos.board, self.pos.hands, self.pos.side_to_move, self.pos.moves = snap
                self.pos._history = []
                self.log_ok("初期局面を読み込みました")
                self.board_view.refresh()
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "clear":
            self.pos.clear_all()
            self.log_ok("盤面・持駒・手順をクリアしました")
            self.board_view.refresh()
            return

        if cmd == "undo":
            try:
                self.pos.undo()
                self.log_ok("undo")
                self.board_view.refresh()
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "sfen" and not arg:
            try:
                snap = self.pos.clone_state()
                s = snapshot_to_sfen(snap)
                self.log_sfen(s)
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd in ("load", "sfen") and arg:
            try:
                snap = sfen_to_snapshot(arg)
                self.pos.board, self.pos.hands, self.pos.side_to_move, self.pos.moves = snap
                self.pos._history = []
                self.log_ok("SFENを読み込みました")
                self.board_view.refresh()
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "kif":
            try:
                text = self._generate_kif_text()
                self.log_kif("出力しました")
                await self.push_screen(KifViewer(text))
            except Exception as e:
                self.log_err(str(e))
            return

        self.log_err("unknown command. type 'help' or enter numeric move.")


if __name__ == "__main__":
    ShogiTui().run()
