#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize, Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Static, RichLog, Input, OptionList

# 既存モジュール
from position import ShogiPosition
from models import Piece
from constants import PIECE_JP, KIND_TO_PYO, PROMOTABLE


# ----------------- Mode -----------------

class Mode(Enum):
    NORMAL = "NORMAL"
    INPUT = "INPUT"


# ----------------- Cursor -----------------

@dataclass(frozen=True)
class Cursor:
    file: int = 5
    rank: int = 5


def _square_label(f: int, r: int) -> str:
    return f"{f}{r}"


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
        return getattr(pos, "hands").get(color, {}) or {}
    if hasattr(pos, "hand") and isinstance(getattr(pos, "hand"), dict):
        return getattr(pos, "hand").get(color, {}) or {}
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
    if tok == ".":
        return None

    color = "B"
    if tok.startswith("v"):
        color = "W"
        tok = tok[1:]

    prom = False
    if tok.startswith("+"):
        prom = True
        tok = tok[1:]

    kind = tok.upper()
    if kind not in PIECE_JP:
        raise ValueError(f"不明な駒トークン: {tok}")

    if prom and (kind not in PROMOTABLE):
        raise ValueError(f"{kind} は成れません")

    return Piece(color=color, kind=kind, prom=prom)


def _build_piece_option_labels(last_token: Optional[str]) -> tuple[list[str], list[str]]:
    """OptionList表示用：labels と tokens を返す（tokens[i] が選択結果）"""
    base = ["P", "L", "N", "S", "G", "B", "R", "K"]
    promo = ["+P", "+L", "+N", "+S", "+B", "+R"]

    tokens: list[str] = []
    labels: list[str] = []

    if last_token:
        tokens.append("__LAST__")
        labels.append(f"Last: {last_token}（直前の駒を再利用）")

    tokens.append(".")
    labels.append(" .  消去")

    # 先手
    for t in base:
        tokens.append(t)
        labels.append(f" {t:<2} 先手 {PIECE_JP[t]}")
    for t in promo:
        kind = t[1:]
        jp = KIND_TO_PYO.get((kind, True), "+" + PIECE_JP[kind])
        tokens.append(t)
        labels.append(f" {t:<2} 先手 {jp}")

    # 後手
    for t in base:
        tokens.append("v" + t)
        labels.append(f" v{t:<2} 後手 {PIECE_JP[t]}")
    for t in promo:
        kind = t[1:]
        jp = KIND_TO_PYO.get((kind, True), "+" + PIECE_JP[kind])
        tokens.append("v" + t)
        labels.append(f" v{t:<2} 後手 {jp}")

    return labels, tokens


# ----------------- Modal: PiecePicker -----------------

class PiecePicker(ModalScreen[Optional[str]]):
    """駒トークンを返すモーダル。キャンセルなら None。ショートカットで高速選択可。"""

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel"),
    ]

    def __init__(self, last_token: Optional[str]):
        super().__init__()
        self.last_token = last_token

        # 状態（ショートカット選択用）
        self.v_on: bool = False
        self.plus_on: bool = False
        self.kind: str = "P"  # P/L/N/S/G/B/R/K

        self._labels, self._tokens = _build_piece_option_labels(last_token)
        self._status: Optional[Static] = None
        self._list: Optional[OptionList] = None

    def compose(self) -> ComposeResult:
        self._status = Static("", id="picker_status")
        yield self._status

        self._list = OptionList(*self._labels, id="picker")
        yield self._list

        hint = Static("keys: v=後手  +=成り  p/l/n/s/g/b/r/k  .=消去  Enter=決定  Esc=cancel", id="picker_hint")
        yield hint

    def on_mount(self) -> None:
        self._update_status()
        # 直前の Enter が OptionSelected に回り込む事故を避ける
        self.set_focus(None)
        self.call_later(self._focus_list)

    def _focus_list(self) -> None:
        if self._list:
            self._list.focus()

    def _update_status(self) -> None:
        if not self._status:
            return
        v = "ON" if self.v_on else "OFF"
        plus = "ON" if self.plus_on else "OFF"
        self._status.update(f"[Picker] v={v} += {plus} kind={self.kind}")

    def _current_token(self) -> str:
        if self.kind == ".":
            return "."
        tok = self.kind
        if self.plus_on and tok in PROMOTABLE:
            tok = "+" + tok
        if self.v_on:
            tok = "v" + tok
        return tok

    def _select_token_in_list(self, tok: str) -> None:
        if not self._list:
            return
        try:
            idx = self._tokens.index(tok)
            self._list.index = idx
        except ValueError:
            pass

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def _confirm(self) -> None:
        self.dismiss(self._current_token())

    def on_key(self, event: Key) -> None:
        k = event.key

        if k == "enter":
            self._confirm()
            event.prevent_default()
            event.stop()
            return

        if k == "v":
            self.v_on = not self.v_on
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k in ("+", "plus"):
            self.plus_on = not self.plus_on
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k == ".":
            self.kind = "."
            self.plus_on = False
            self.v_on = False
            self._update_status()
            self._select_token_in_list(".")
            event.prevent_default()
            event.stop()
            return

        if k == "0" and self.last_token:
            self.dismiss(self.last_token)
            event.prevent_default()
            event.stop()
            return

        m = {"p": "P", "l": "L", "n": "N", "s": "S", "g": "G", "b": "B", "r": "R", "k": "K"}
        if k in m:
            self.kind = m[k]
            if self.plus_on and self.kind not in PROMOTABLE:
                self.plus_on = False
            self._update_status()
            self._select_token_in_list(self._current_token())
            event.prevent_default()
            event.stop()
            return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        tok = self._tokens[event.option_index]
        if tok == "__LAST__":
            self.dismiss(self.last_token)
            return
        self.dismiss(tok)


# ----------------- Modal: HandPicker -----------------

class HandPicker(ModalScreen[Optional[tuple[str, str, int]]]):
    """
    (color, kind, n) を返す。キャンセルなら None。
    重要: Input を使わない（キーが吸われて駒種変更できない問題の根本解決）

    keys:
      v=先後トグル
      p/l/n/s/g/b/r = 駒種
      0-9 = 枚数入力（追記）
      Backspace = 1文字消し
      c = クリア
      Enter = 確定
      Esc = cancel
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel"),
    ]

    def __init__(self, default_color: str = "B"):
        super().__init__()
        self.color = default_color  # "B" or "W"
        self.kind: str = "P"
        self._digits: str = "1"

        self._status: Optional[Static] = None
        self._hint: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._status = Static("", id="hand_status")
        yield self._status

        self._hint = Static(
            "keys: v=先後  p/l/n/s/g/b/r=駒種  0-9=枚数  BS=削除  c=clear  Enter=確定  Esc=cancel",
            id="hand_hint",
        )
        yield self._hint

    def on_mount(self) -> None:
        self._update_status()
        # フォーカスをScreenに置く（常に on_key で拾う）
        self.set_focus(None)

    def _update_status(self) -> None:
        if not self._status:
            return
        side = "先手" if self.color == "B" else "後手"
        n = self._digits if self._digits else "0"
        self._status.update(f"[Hand] {side} kind={self.kind} n={n}")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def _current_n(self) -> int:
        s = self._digits.strip() if self._digits else "0"
        try:
            n = int(s)
        except Exception:
            n = 0
        return max(0, n)

    def on_key(self, event: Key) -> None:
        k = event.key

        if k == "v":
            self.color = "W" if self.color == "B" else "B"
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        m = {"p": "P", "l": "L", "n": "N", "s": "S", "g": "G", "b": "B", "r": "R"}
        if k in m:
            self.kind = m[k]
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k in ("backspace", "delete"):
            self._digits = self._digits[:-1]
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k == "c":
            self._digits = ""
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k.isdigit():
            # 先頭ゼロは抑制したいならここで調整（今は素直に追記）
            self._digits = (self._digits + k) if self._digits is not None else k
            # 無駄に長くならないよう軽く制限（3桁まで）
            if len(self._digits) > 3:
                self._digits = self._digits[-3:]
            self._update_status()
            event.prevent_default()
            event.stop()
            return

        if k == "enter":
            n = self._current_n()
            self.dismiss((self.color, self.kind, n))
            event.prevent_default()
            event.stop()
            return


# ----------------- Widgets -----------------

class BoardView(Static):
    """盤面表示（カーソル位置をハイライト + 持駒表示）"""
    cursor: Cursor = reactive(Cursor(5, 5))

    def __init__(self, pos: ShogiPosition, **kwargs):
        super().__init__(**kwargs)
        self.pos = pos

    def render(self) -> Text:
        t = Text()

        # 上：後手持駒
        gote_hand = _format_hand(self.pos, "W")
        t.append(f"▽持駒: {gote_hand}\n")

        # 盤
        t.append("  ９ ８ ７ ６ ５ ４ ３ ２ １\n")
        t.append("+---------------------------+\n")
        for r in range(1, 10):
            t.append("|")
            for f in range(9, 0, -1):
                cell = _cell_str(self.pos, f, r)
                is_cursor = (f == self.cursor.file and r == self.cursor.rank)
                t.append(cell, style="reverse" if is_cursor else "")
            t.append(f"|{r}\n")
        t.append("+---------------------------+\n")

        # 下：先手持駒（要望）
        sente_hand = _format_hand(self.pos, "B")
        t.append(f"▲持駒: {sente_hand}\n")

        return t


class HelpPanel(Static):
    def render(self) -> Text:
        txt = Text()
        txt.append("駒トークン:\n", style="bold")
        txt.append("  P=歩  L=香  N=桂  S=銀  G=金  B=角  R=飛  K=玉\n")
        txt.append("  +P=と  +L=成香  +N=成桂  +S=成銀  +B=馬  +R=竜\n")
        txt.append("  vP=後手歩  v+R=後手竜  .=消去\n\n")

        txt.append("キー（NORMAL）:\n", style="bold")
        txt.append("  h/j/k/l または 矢印: カーソル移動\n")
        txt.append("  Enter: 駒配置ピッカー\n")
        txt.append("  x: 消去  /  g: 55へ戻る\n")
        txt.append("  H: 持ち駒ピッカー\n")
        txt.append("  ?: ヘルプ表示トグル\n")
        txt.append("  i / : : INPUTへ  /  Esc: NORMALへ\n\n")

        txt.append("終了:\n", style="bold")
        txt.append("  Ctrl+C: 終了\n")
        txt.append("  q: NORMAL中のみ終了（INPUT中は無効、:q 推奨）\n")
        return txt


# ----------------- App -----------------

class ShogiTui(App):
    CSS = """
    Screen { padding: 0; }
    #root { height: 100%; padding: 0; }

    /* 盤面領域（盤面+右help）: 持駒2行ぶん増えるので 14 行確保 */
    #top {
        height: 14;
        min-height: 14;
        padding: 0;
        margin: 0;
    }

    #board_col {
        width: 34;
        min-width: 34;
        height: 14;
        min-height: 14;
        padding: 0 1;
        margin: 0;
    }

    #side_col {
        width: 1fr;
        min-width: 18;
        height: 14;
        min-height: 14;
        padding: 0;
        margin: 0;
    }

    /* 狭い時に盤面の下に出すHelp（普段は隠す） */
    #help_bottom {
        display: none;
        height: auto;
        max-height: 8;
        padding: 0 1;
        margin: 0;
        border: none;
    }

    /* 下段は残り全部 */
    #bottom {
        height: 1fr;
        min-height: 5;
        padding: 0;
        margin: 0;
    }

    #status {
        height: 1;
        padding: 0 1;
        margin: 0;
    }

    /* Inputは3行にしてカーソル/枠が潰れないように */
    #cmd {
        height: 3;
        min-height: 3;
        padding: 0 1;
        margin: 0;
    }

    #log {
        height: 1fr;
        padding: 0 1;
        margin: 0;
        overflow: auto;
        scrollbar-size: 1 1;
    }

    BoardView { height: 14; }
    BoardView, #board_col * { text-wrap: nowrap; }

    /* Picker */
    #picker_status { height: 1; padding: 0 1; }
    #picker_hint { height: 1; padding: 0 1; }

    #hand_status { height: 1; padding: 0 1; }
    #hand_hint   { height: 1; padding: 0 1; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("?", "toggle_help", "Help"),

        Binding("i", "enter_input", "Input"),
        Binding(":", "enter_input", "Command"),
        Binding("escape", "leave_input", "Normal"),

        Binding("x", "clear_square", "Clear"),
        Binding("g", "goto_home", "Home(55)"),
        Binding("H", "open_hand_picker", "Hand"),

        # q は残すが、action_quit側で「NORMALのみ」終了に制御する
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.pos = ShogiPosition()
        self.cursor = Cursor(5, 5)

        self.mode: Mode = Mode.INPUT
        self.help_visible: bool = True
        self.last_piece_token: Optional[str] = None

        # 次に置く駒用（ステータス表示のため）
        self.next_v_on: bool = False
        self.next_plus_on: bool = False

        # ピッカーを開いた瞬間のマス（ズレ防止）
        self._pending_square: Optional[Tuple[int, int]] = None

        self.board_view: Optional[BoardView] = None
        self.log_widget: Optional[RichLog] = None
        self.cmd: Optional[Input] = None
        self.status: Optional[Static] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Horizontal(id="top"):
                with Vertical(id="board_col"):
                    self.board_view = BoardView(self.pos)
                    yield self.board_view

                with VerticalScroll(id="side_col"):
                    yield HelpPanel(id="help_panel_right")

            with VerticalScroll(id="help_bottom"):
                yield HelpPanel(id="help_panel_bottom")

            with Vertical(id="bottom"):
                self.status = Static("", id="status")
                yield self.status

                self.cmd = Input(
                    placeholder="コマンド入力（例: p 55 P / h b G 1 / show / :q）",
                    id="cmd",
                )
                yield self.cmd

                self.log_widget = RichLog(id="log", wrap=False, highlight=True)
                yield self.log_widget

    def on_mount(self) -> None:
        self._sync_cursor()
        self._set_status()

        if self.log_widget:
            self.log_widget.write("起動しました。EscでNORMAL、i/:(コロン)でINPUT。Enterで駒ピッカー。")

        if self.cmd:
            self.cmd.focus()

        self._apply_help_layout(self.size.width, self.size.height)

    # ---- Core helpers ----

    def _refresh_board(self) -> None:
        if self.board_view:
            self.board_view.refresh()

    def _sync_cursor(self) -> None:
        if self.board_view:
            self.board_view.cursor = self.cursor

    def _set_status(self) -> None:
        if not self.status:
            return
        v = "ON" if self.next_v_on else "OFF"
        plus = "ON" if self.next_plus_on else "OFF"
        sente_hand = _format_hand(self.pos, "B")
        self.status.update(
            f"[{self.mode.value}] "
            f"sq={_square_label(self.cursor.file, self.cursor.rank)} "
            f"turn={self.pos.side_to_move} "
            f"v={v} += {plus} "
            f"| ▲持駒:{sente_hand}"
        )

    def _move_cursor(self, df: int, dr: int) -> None:
        f = min(9, max(1, self.cursor.file + df))
        r = min(9, max(1, self.cursor.rank + dr))
        self.cursor = Cursor(f, r)
        self._sync_cursor()
        self._set_status()

    # ---- Layout / Help toggle ----

    def on_resize(self, event: Resize) -> None:
        self._apply_help_layout(event.size.width, event.size.height)

    def _apply_help_layout(self, w: int, h: int) -> None:
        board_col = self.query_one("#board_col")
        side_col = self.query_one("#side_col")
        help_bottom = self.query_one("#help_bottom")

        if not self.help_visible:
            side_col.styles.display = "none"
            help_bottom.styles.display = "none"
            board_col.styles.width = "100%"
            board_col.styles.min_width = 0
            return

        wide = (w >= 70 and h >= 18)  # 14行になったので少し上げる
        if wide:
            side_col.styles.display = "block"
            help_bottom.styles.display = "none"
            board_col.styles.width = 34
            board_col.styles.min_width = 34
            side_col.styles.width = "1fr"
            side_col.styles.min_width = 18
        else:
            side_col.styles.display = "none"
            help_bottom.styles.display = "block"
            board_col.styles.width = "100%"
            board_col.styles.min_width = 0

    def action_toggle_help(self) -> None:
        self.help_visible = not self.help_visible
        self._apply_help_layout(self.size.width, self.size.height)

    # ---- Mode switching ----

    def action_enter_input(self) -> None:
        self.mode = Mode.INPUT
        if self.cmd:
            self.cmd.focus()
        self._set_status()

    def action_leave_input(self) -> None:
        self.mode = Mode.NORMAL
        self.set_focus(None)
        self._set_status()

    # ---- Quit behavior (NORMAL only) ----

    async def action_quit(self) -> None:
        if self.mode == Mode.INPUT:
            if self.log_widget:
                self.log_widget.write("[hint] INPUT中は q では終了しません。:q を入力して Enter（または Ctrl+C）")
            return
        await super().action_quit()

    # ---- Misc actions ----

    def action_goto_home(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor = Cursor(5, 5)
        self._sync_cursor()
        self._set_status()

    def action_clear_square(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        sq = (self.cursor.file, self.cursor.rank)
        self.pos.set_piece(sq, None)
        self._refresh_board()
        self._set_status()
        if self.log_widget:
            self.log_widget.write(f"[p] {_square_label(*sq)} <- 消去")

    # ---- Piece picker ----

    def _open_piece_picker(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self._pending_square = (self.cursor.file, self.cursor.rank)
        self.push_screen(PiecePicker(self.last_piece_token), callback=self._on_piece_picked)

    def _on_piece_picked(self, tok: Optional[str]) -> None:
        if not tok:
            self._pending_square = None
            return

        if tok == "__LAST__":
            tok = self.last_piece_token

        if not tok:
            self._pending_square = None
            return

        try:
            p = parse_piece_token(tok)

            sq = self._pending_square or (self.cursor.file, self.cursor.rank)
            self._pending_square = None

            self.pos.set_piece(sq, p)

            if tok != ".":
                self.last_piece_token = tok

            self.next_v_on = tok.startswith("v")
            self.next_plus_on = (tok.startswith("v+") or tok.startswith("+"))

            self._refresh_board()
            self._set_status()

            if self.log_widget:
                placed = "消去" if p is None else f"{tok}"
                self.log_widget.write(f"[p] {_square_label(*sq)} <- {placed}")

        except Exception as e:
            self._pending_square = None
            if self.log_widget:
                self.log_widget.write(f"[p] エラー: {e}")

    # ---- Hand picker ----

    def action_open_hand_picker(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.push_screen(HandPicker(default_color="B"), callback=self._on_hand_picked)

    def _on_hand_picked(self, result: Optional[tuple[str, str, int]]) -> None:
        if not result:
            return
        color, kind, n = result
        try:
            self.pos.set_hand(color, kind, n)
            # 持駒が図にもステータスにも出るので両方更新
            self._refresh_board()
            self._set_status()
            if self.log_widget:
                side = "b" if color == "B" else "w"
                self.log_widget.write(f"[h] {side} {kind} {n}")
        except Exception as e:
            if self.log_widget:
                self.log_widget.write(f"[h] エラー: {e}")

    # ---- Key handling ----

    async def on_key(self, event: Key) -> None:
        if self.mode == Mode.NORMAL:
            if event.key in ("h", "left"):
                self._move_cursor(+1, 0)
                event.stop()
                return
            if event.key in ("l", "right"):
                self._move_cursor(-1, 0)
                event.stop()
                return
            if event.key in ("k", "up"):
                self._move_cursor(0, -1)
                event.stop()
                return
            if event.key in ("j", "down"):
                self._move_cursor(0, +1)
                event.stop()
                return

            if event.key == "enter":
                # Enter がピッカー側へ回り込むのを止める
                event.prevent_default()
                event.stop()
                self.call_later(self._open_piece_picker)
                return

    # ---- Command input ----

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""

        if not line:
            return

        if self.log_widget:
            self.log_widget.write(f"> {line}")

        if line in (":q", ":quit"):
            self.mode = Mode.NORMAL
            await super().action_quit()
            return

        if line in ("q", "quit", "exit"):
            if self.log_widget:
                self.log_widget.write("[hint] INPUT中の q は終了しません。:q を入力してください（または Ctrl+C）")
            return

        if line == "show":
            if self.log_widget:
                self.log_widget.write(f"手番={self.pos.side_to_move}, moves={len(self.pos.moves)}")
            return

        if self.log_widget:
            self.log_widget.write("（未接続）このコマンドは次の段階で既存CLIロジックに繋ぎます。")


if __name__ == "__main__":
    ShogiTui().run()
