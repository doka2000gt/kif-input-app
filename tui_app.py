#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.reactive import reactive
from textual.widgets import Static, RichLog, Input

# 既存モジュールを流用
from position import ShogiPosition
from constants import PIECE_JP, KIND_TO_PYO


# ----------------- Vim-like modes -----------------

class Mode(Enum):
    NORMAL = "NORMAL"
    INPUT = "INPUT"


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


class BoardView(Static):
    """盤面表示（カーソル位置をハイライトするだけの最小Widget）"""
    cursor: Cursor = reactive(Cursor(5, 5))

    def __init__(self, pos: ShogiPosition, **kwargs):
        super().__init__(**kwargs)
        self.pos = pos

    def render(self) -> Text:
        t = Text()
        t.append("  ９ ８ ７ ６ ５ ４ ３ ２ １\n")
        t.append("+---------------------------+\n")
        for r in range(1, 10):
            t.append("|")
            for f in range(9, 0, -1):
                cell = _cell_str(self.pos, f, r)
                is_cursor = (f == self.cursor.file and r == self.cursor.rank)
                t.append(cell, style="reverse" if is_cursor else None)
            t.append(f"|{r}\n")
        t.append("+---------------------------+\n")
        return t


class HelpPanel(Static):
    def render(self) -> Text:
        txt = Text()
        txt.append("駒トークン:\n", style="bold")
        txt.append("  P=歩  L=香  N=桂  S=銀  G=金  B=角  R=飛  K=玉\n")
        txt.append("  +P=と  +L=成香  +N=成桂  +S=成銀  +B=馬  +R=竜\n")
        txt.append("  vP=後手歩  v+R=後手竜  .=消去\n")
        txt.append("\n操作:\n", style="bold")
        txt.append("  NORMAL: hjkl/矢印=移動  i or :=入力  ?=ヘルプ  q=終了\n")
        txt.append("  INPUT : Enter=実行  Esc=戻る\n")
        return txt


class ShogiTui(App):
    CSS = """
    Screen { padding: 0; }

    #root { height: 100%; padding: 0; }

    /* 上段：盤面＋（広い時は）右ヘルプ */
    #top {
        height: 12;
        min-height: 12;
        padding: 0;
        margin: 0;
    }

    #board_col {
        width: 34;
        min-width: 34;
        height: 12;
        min-height: 12;
        padding: 0 1;
        margin: 0;
    }

    #side_col {
        width: 1fr;
        min-width: 18;
        height: 12;
        min-height: 12;
        padding: 0;
        margin: 0;
    }

    /* 狭い時の「盤面下ヘルプ」（普段は隠す） */
    #help_bottom {
        display: none;
        height: auto;
        max-height: 8;
        padding: 0 1;
        margin: 0;
    }

    /* 下段：常に見せる */
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

    /* Input は 3 行にしてカーソルが枠に隠れないように */
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

    BoardView { height: 12; }
    BoardView, #board_col * { text-wrap: nowrap; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("?", "toggle_help", "Help"),
        ("i", "enter_input", "Input"),
        (":", "enter_input", "Command"),
        ("escape", "leave_input", "Normal"),
    ]

    def __init__(self):
        super().__init__()
        self.pos = ShogiPosition()
        self.cursor = Cursor(5, 5)

        self.mode: Mode = Mode.INPUT  # 起動時は入力
        self.help_visible: bool = True

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

                # 右ヘルプ（広い時）
                with VerticalScroll(id="side_col"):
                    yield HelpPanel(id="help_panel_right")

            # 下ヘルプ（狭い時）
            with VerticalScroll(id="help_bottom"):
                yield HelpPanel(id="help_panel_bottom")

            with Vertical(id="bottom"):
                self.status = Static("", id="status")
                yield self.status

                self.cmd = Input(
                    placeholder="コマンド入力（例: p 55 P / h b G 1 / show）",
                    id="cmd",
                )
                yield self.cmd

                self.log_widget = RichLog(id="log", wrap=False, highlight=True)
                yield self.log_widget

    def on_mount(self) -> None:
        self._sync_cursor()
        self._set_status()
        if self.log_widget:
            self.log_widget.write("起動しました。INPUT: Enter実行 / EscでNORMAL。NORMAL: hjkl/矢印移動、?でヘルプ。")

        if self.cmd:
            self.cmd.focus()

        # 初回レイアウト確定
        self.call_after_refresh(self._apply_help_layout)

    def on_resize(self, event: Resize) -> None:
        self._apply_help_layout()

    # ----------------- Layout helpers -----------------

    def _apply_help_layout(self) -> None:
        """幅が狭い時はヘルプを下へ。? トグルで完全に消える。"""
        board_col = self.query_one("#board_col")
        side_col = self.query_one("#side_col")
        help_bottom = self.query_one("#help_bottom")

        w = self.size.width
        h = self.size.height

        wide = (w >= 70 and h >= 16)

        if not self.help_visible:
            side_col.styles.display = "none"
            help_bottom.styles.display = "none"
            board_col.styles.width = "100%"
            board_col.styles.min_width = 0
            return

        if wide:
            # 右に出す
            side_col.styles.display = "block"
            help_bottom.styles.display = "none"
            board_col.styles.width = 34
            board_col.styles.min_width = 34
            side_col.styles.width = "1fr"
            side_col.styles.min_width = 18
        else:
            # 下に出す
            side_col.styles.display = "none"
            help_bottom.styles.display = "block"
            board_col.styles.width = "100%"
            board_col.styles.min_width = 0

    # ----------------- Actions -----------------

    def action_toggle_help(self) -> None:
        self.help_visible = not self.help_visible
        self._apply_help_layout()

    def action_enter_input(self) -> None:
        self.mode = Mode.INPUT
        if self.cmd:
            self.cmd.focus()
        self._set_status()

    def action_leave_input(self) -> None:
        self.mode = Mode.NORMAL
        # フォーカスを外す（ここが「矢印がInputに取られる」問題の本体）
        self.set_focus(None)
        self._set_status()

    # ----------------- Cursor & status -----------------

    def _sync_cursor(self) -> None:
        if self.board_view:
            self.board_view.cursor = self.cursor

    def _set_status(self) -> None:
        if not self.status:
            return
        self.status.update(
            f"[{self.mode.value}] "
            f"選択マス: {_square_label(self.cursor.file, self.cursor.rank)} / "
            f"手番: {self.pos.side_to_move}（B=先手, W=後手）"
        )

    def _move_cursor(self, df: int, dr: int) -> None:
        f = min(9, max(1, self.cursor.file + df))
        r = min(9, max(1, self.cursor.rank + dr))
        self.cursor = Cursor(f, r)
        self._sync_cursor()
        self._set_status()

    async def on_key(self, event) -> None:
        # NORMAL のときだけ盤面操作
        if self.mode != Mode.NORMAL:
            return

        # hjkl + 矢印（保険）
        if event.key in ("h", "left"):
            self._move_cursor(+1, 0)
        elif event.key in ("l", "right"):
            self._move_cursor(-1, 0)
        elif event.key in ("k", "up"):
            self._move_cursor(0, -1)
        elif event.key in ("j", "down"):
            self._move_cursor(0, +1)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""

        if not line:
            return

        if self.log_widget:
            self.log_widget.write(f"> {line}")

        # 最小例
        if line == "show":
            if self.log_widget:
                self.log_widget.write(f"手番={self.pos.side_to_move}, moves={len(self.pos.moves)}")
            return

        if line in ("q", "quit", "exit"):
            await self.action_quit()
            return

        if self.log_widget:
            self.log_widget.write("（未接続）このコマンドは次の段階で既存CLIロジックに繋ぎます。")


if __name__ == "__main__":
    ShogiTui().run()
