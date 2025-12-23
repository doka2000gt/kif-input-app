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
from textual.widgets import Static, RichLog, Input, OptionList

# 既存モジュール（プロジェクト側にある前提）
from position import ShogiPosition
from models import Piece
from constants import PIECE_JP, KIND_TO_PYO, PROMOTABLE
from sfen import snapshot_to_sfen, sfen_to_snapshot
from manual_kif import kif_line_for_minimal_move


# ----------------- Mode -----------------

class Mode(Enum):
    NORMAL = "NORMAL"
    INPUT = "INPUT"


# ----------------- Cursor -----------------

@dataclass
class Cursor:
    file: int = 5
    rank: int = 5


def _square_label(f: int, r: int) -> str:
    return f"{f}{r}"


def _get_hand_dict(pos: ShogiPosition, color: str) -> Dict[str, int]:
    """pos.hands / pos.hand のどちらでも動くように吸収。想定: {"P":3, "R":1, ...}"""
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


def _cell_str(pos: ShogiPosition, f: int, r: int) -> str:
    """盤面セルの文字列（2文字想定：' ・' / ' 歩' / 'v歩' / 'v竜' など）"""
    p = pos.board.get((f, r))
    if p is None:
        return " ・"
    name = KIND_TO_PYO.get((p.kind, bool(getattr(p, "prom", False))), PIECE_JP.get(p.kind, p.kind))
    return ("v" + name) if p.color == "W" else (" " + name)


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
    if tok.lower().startswith("v"):
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

    # ★ここが重要：models.Piece を返す（prom 属性あり）
    return Piece(color=color, kind=kind, prom=prom)


# ----------------- Modal: PiecePicker (OptionList) -----------------

def _build_piece_option_labels(last_token: Optional[str]) -> tuple[list[str], list[str]]:
    """OptionList表示用：labels と tokens を返す（tokens[i] が選択結果）"""
    base = ["P", "L", "N", "S", "G", "B", "R", "K"]
    promo = ["+P", "+L", "+N", "+S", "+B", "+R"]

    tokens: list[str] = []
    labels: list[str] = []

    if last_token:
        tokens.append("__LAST__")
        labels.append(f"直前: {last_token}（直前のトークンを再利用）")

    tokens.append(".")
    labels.append(" .   消去")

    # 先手
    for t in base:
        tokens.append(t)
        labels.append(f" {t:<2} 先手  {PIECE_JP[t]}")
    for t in promo:
        kind = t[1:]
        jp = KIND_TO_PYO.get((kind, True), "+" + PIECE_JP[kind])
        tokens.append(t)
        labels.append(f" {t:<2} 先手  {jp}")

    # 後手
    for t in base:
        tokens.append("v" + t)
        labels.append(f" v{t:<2} 後手  {PIECE_JP[t]}")
    for t in promo:
        kind = t[1:]
        jp = KIND_TO_PYO.get((kind, True), "+" + PIECE_JP[kind])
        tokens.append("v" + t)
        labels.append(f" v{t:<2} 後手  {jp}")

    return labels, tokens


class PiecePicker(ModalScreen[Optional[str]]):
    """駒トークンを返すモーダル。Esc/qでキャンセル。"""

    BINDINGS = [
        Binding("escape", "dismiss_none", "キャンセル"),
        Binding("q", "dismiss_none", "キャンセル"),
    ]

    def __init__(self, last_token: Optional[str]):
        super().__init__()
        self.last_token = last_token
        self._labels, self._tokens = _build_piece_option_labels(last_token)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("駒配置：一覧から選ぶ（Enter） / キー直接入力も可（P/L/N/S/G/B/R/K, v, +）", classes="modal_title"),
            OptionList(*self._labels, id="picker"),
            classes="modal_box",
        )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def _dismiss_by_index(self, idx: int) -> None:
        tok = self._tokens[idx]
        if tok == "__LAST__":
            tok = self.last_token or None
        self.dismiss(tok)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._dismiss_by_index(event.option_index)

    def on_key(self, event: Key) -> None:
        # Enterで決定
        if event.key == "enter":
            ol = self.query_one(OptionList)
            if ol.highlighted is not None:
                self._dismiss_by_index(ol.highlighted)
                event.prevent_default()
                event.stop()
            return

        # 直接入力ショートカット（例: v + P）
        # v で後手トグル、+ で成りトグル、駒種で即決
        k = event.key
        if k in ("v", "V"):
            # v トグル（ただし “v”単体で確定しない）
            # 何もしない（表示更新等は省略）
            event.prevent_default()
            event.stop()
            return
        if k == "+":
            event.prevent_default()
            event.stop()
            return

        kk = k.upper()
        if kk in ("P", "L", "N", "S", "G", "B", "R", "K"):
            # ここは「最後に押したキーだけ」で確定せず、
            # まずは最小限のトークンにする（P等）。
            # v/+ を組み合わせたい場合は INPUT コマンド p で入れてもOK。
            self.dismiss(kk)
            event.prevent_default()
            event.stop()
            return


# ----------------- Modal: DropPicker -----------------

class DropPicker(ModalScreen[Optional[str]]):
    """打ちの駒種を選ぶモーダル。候補(kind文字)を返す。キャンセルなら None。"""

    BINDINGS = [
        Binding("escape", "dismiss_none", "キャンセル"),
        Binding("q", "dismiss_none", "キャンセル"),
    ]

    def __init__(self, candidates: list[str], to_sq: tuple[int, int]):
        super().__init__()
        self.candidates = candidates
        self.to_sq = to_sq

    def compose(self) -> ComposeResult:
        title = Static(f"打ち：{self.to_sq[0]}{self.to_sq[1]} に打つ駒を選ぶ（Enter）", classes="modal_title")
        opts = OptionList(*[f"{k}:{PIECE_JP.get(k, k)}" for k in self.candidates], id="drop_options")
        yield Vertical(title, opts, classes="modal_box")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        kk = event.key.upper()
        if kk in self.candidates:
            self.dismiss(kk)
            event.prevent_default()
            event.stop()
            return
        if event.key == "enter":
            ol = self.query_one(OptionList)
            if ol.highlighted is not None:
                label = ol.get_option_at_index(ol.highlighted).prompt.plain
                kind = label.split(":", 1)[0]
                self.dismiss(kind)
                event.prevent_default()
                event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        label = event.option.prompt.plain
        kind = label.split(":", 1)[0]
        self.dismiss(kind)


# ----------------- Modal: HandPicker -----------------

class HandPicker(ModalScreen[Optional[tuple[str, str, int]]]):
    """持駒を (color, kind, n) で返す。"""

    BINDINGS = [
        Binding("escape", "dismiss_none", "キャンセル"),
        Binding("q", "dismiss_none", "キャンセル"),
    ]

    def __init__(self, default_color: str = "B"):
        super().__init__()
        self.default_color = default_color
        self.color = default_color
        self.kind = "P"
        self.n = 1

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("持駒設定： b/w（先手/後手）, 駒種(P/L/N/S/G/B/R/K), 枚数(1..)", classes="modal_title"),
            Static("例： h b G 1   （先手に金を1枚）", classes="modal_hint"),
            Input(value=f"{'b' if self.default_color=='B' else 'w'} P 1", id="hand_input"),
            classes="modal_box",
        )

    def on_mount(self) -> None:
        self.query_one("#hand_input", Input).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        s = event.value.strip()
        m = re.fullmatch(r"(b|w)\s+([PLNSGBRK])\s+(\d+)", s, re.IGNORECASE)
        if not m:
            self.dismiss(None)
            return
        side = m.group(1).lower()
        kind = m.group(2).upper()
        n = int(m.group(3))
        color = "B" if side == "b" else "W"
        self.dismiss((color, kind, n))


# ----------------- Modal: KifViewer -----------------

class KifViewer(ModalScreen[None]):
    """KIFテキストを表示するだけのモーダル。Esc/qで閉じる。"""

    BINDINGS = [
        Binding("escape", "close", "閉じる"),
        Binding("q", "close", "閉じる"),
    ]

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("KIF出力（Esc / q で閉じる）", classes="modal_title"),
            VerticalScroll(Static(self.text, classes="kif_text"), classes="modal_box"),
        )

    def action_close(self) -> None:
        self.dismiss(None)


# ----------------- Views -----------------

class BoardView(Static):
    """盤面表示（カーソル位置をハイライト + 持駒表示）"""
    cursor: Cursor = reactive(Cursor(5, 5))

    def __init__(self, pos: ShogiPosition, cursor: Cursor, **kwargs):
        super().__init__(**kwargs)
        self.pos = pos
        self.cursor = cursor

    def render(self) -> Text:
        t = Text()

        # 上：後手持駒
        t.append(f"▽持駒: {_format_hand(self.pos, 'W')}\n")

        # 盤面（上から1段目）
        for r in range(1, 10):
            for f in range(9, 0, -1):
                cell = _cell_str(self.pos, f, r)
                if (f == self.cursor.file and r == self.cursor.rank):
                    t.append(cell, style="reverse")
                else:
                    t.append(cell)
            t.append("\n")

        # 下：先手持駒
        t.append(f"△持駒: {_format_hand(self.pos, 'B')}\n")
        return t


class HelpView(Static):
    def render(self) -> str:
        return (
            "[b]操作[/b]\n"
            "\n"
            "[b]NORMAL[/b]\n"
            "  hjkl / 矢印  : カーソル移動\n"
            "  Enter        : そのマスの駒ピッカー\n"
            "  p            : 駒ピッカー（直前トークン）\n"
            "  h            : 持駒ピッカー\n"
            "  x            : そのマスを消去\n"
            "  i / :        : INPUTへ\n"
            "  q            : 終了\n"
            "\n"
            "[b]INPUT[/b]（入力して Enter）\n"
            "  :q           : 終了\n"
            "  help / ?     : ヘルプ\n"
            "  show         : 状態表示\n"
            "  sfen         : 現局面をSFEN表示\n"
            "  load <SFEN>  : SFEN読み込み\n"
            "  clear        : 盤面・持駒・手順クリア\n"
            "  undo         : 1手戻す\n"
            "  start        : 初期局面へ\n"
            "  p <sq> <tok> : 駒配置（例: p 55 P / p 33 vK / p 11 .）\n"
            "  h <b|w> <K> <N> : 持駒（例: h b G 1）\n"
            "  7776 / 77761 / 076 : 数字入力（移動/成り/打ち）\n"
            "  kif          : KIF表示\n"
            "\n"
            "  Esc          : NORMALへ戻る\n"
        )


# ----------------- App -----------------

class ShogiTui(App):
    CSS = """
    Screen { layout: vertical; }
    #root { height: 1fr; layout: horizontal; }
    #board_col { width: 36; min-width: 36; height: 1fr; }
    #side_col { width: 1fr; min-width: 20; height: 1fr; }

    #board { height: auto; border: solid gray; padding: 1 1; }
    #help  { height: auto; border: solid gray; padding: 1 1; }
    #help_bottom { height: auto; border: solid gray; padding: 1 1; }

    #cmd { height: 3; border: solid gray; padding: 0 1; }
    #log { height: 1fr; border: solid gray; padding: 1 1; }

    .modal_box {
        width: 90%;
        height: auto;
        border: solid white;
        padding: 1 2;
        background: black;
    }
    .modal_title { margin: 0 0 1 0; }
    .modal_hint { color: gray; margin: 0 0 1 0; }
    """

    BINDINGS = [
        ("q", "quit", "終了"),
        ("i", "enter_input", "INPUT"),
        (":", "enter_input", "INPUT"),
        ("escape", "leave_input", "NORMAL"),
        ("h", "open_hand_picker", "持駒"),
        ("p", "open_piece_picker", "駒"),
        ("x", "clear_square", "消去"),
        ("?", "toggle_help", "ヘルプ"),
    ]

    def __init__(self):
        super().__init__()
        self.pos = ShogiPosition()
        self.cursor = Cursor(5, 5)
        self.mode: Mode = Mode.INPUT

        self.last_piece_token: str = "P"
        self._pending_square: Optional[tuple[int, int]] = None

        self.cmd: Optional[Input] = None
        self.status: Optional[Static] = None
        self.board_view: Optional[BoardView] = None
        self.help_view: Optional[HelpView] = None
        self.help_bottom: Optional[HelpView] = None
        self.log_widget: Optional[RichLog] = None
        self.help_visible: bool = True

    def compose(self) -> ComposeResult:
        self.status = Static("", id="status")
        self.board_view = BoardView(self.pos, self.cursor, id="board")
        self.help_view = HelpView(id="help")
        self.help_bottom = HelpView(id="help_bottom")
        self.cmd = Input(placeholder="コマンド / 指し手（Enterで実行）", id="cmd")
        self.log_widget = RichLog(id="log", wrap=False, highlight=True)

        with Vertical():
            yield self.status
            with Horizontal(id="root"):
                with Vertical(id="board_col"):
                    yield self.board_view
                with Vertical(id="side_col"):
                    yield self.help_view
            yield self.help_bottom
            yield self.cmd
            yield self.log_widget

    def on_mount(self) -> None:
        self._set_status()
        if self.log_widget:
            self.log_widget.write("起動しました。EscでNORMAL、i/:(コロン)でINPUT。NORMALでEnter=駒ピッカー。")
        if self.cmd:
            self.cmd.focus()
        self._apply_help_layout(self.size.width, self.size.height)

    def on_resize(self, event: Resize) -> None:
        self._apply_help_layout(event.size.width, event.size.height)

    def _apply_help_layout(self, w: int, h: int) -> None:
        """横幅が広い時は右にヘルプ、狭い時は下にヘルプ。"""
        if not self.help_view or not self.help_bottom:
            return
        if not self.help_visible:
            self.help_view.styles.display = "none"
            self.help_bottom.styles.display = "none"
            return

        board_col = self.query_one("#board_col", Vertical)
        side_col = self.query_one("#side_col", Vertical)

        wide = (w >= 70 and h >= 18)
        if wide:
            side_col.styles.display = "block"
            self.help_bottom.styles.display = "none"
            board_col.styles.width = 36
            board_col.styles.min_width = 36
            side_col.styles.width = "1fr"
            side_col.styles.min_width = 20
        else:
            side_col.styles.display = "none"
            self.help_bottom.styles.display = "block"
            board_col.styles.width = "100%"
            board_col.styles.min_width = 0

    def action_toggle_help(self) -> None:
        self.help_visible = not self.help_visible
        self._apply_help_layout(self.size.width, self.size.height)

    # ---- モード切替 ----

    def action_enter_input(self) -> None:
        self.mode = Mode.INPUT
        if self.cmd:
            self.cmd.focus()
        self._set_status()

    def action_leave_input(self) -> None:
        self.mode = Mode.NORMAL
        self.set_focus(None)
        self._set_status()

    # ---- 状態 ----

    def _set_status(self) -> None:
        if not self.status:
            return
        mode = "NORMAL" if self.mode == Mode.NORMAL else "INPUT"
        self.status.update(
            f"mode={mode}  cursor={self.cursor.file}{self.cursor.rank}  "
            f"turn={getattr(self.pos, 'side_to_move', '?')}  "
            f"moves={len(getattr(self.pos, 'moves', []) or [])}"
        )

    def _refresh(self) -> None:
        if self.board_view:
            self.board_view.refresh()
        if self.help_view:
            self.help_view.refresh()
        if self.help_bottom:
            self.help_bottom.refresh()

    # ---- undo用（pos側が未対応でも動くように） ----

    def _push_history(self) -> None:
        try:
            snap = self.pos.clone_state()
        except Exception:
            return
        try:
            if not hasattr(self.pos, "_history") or self.pos._history is None:
                self.pos._history = []
            self.pos._history.append(snap)
        except Exception:
            return

    # ---- KIF ----

    def _generate_kif_text(self) -> str:
        lines: list[str] = []
        lines.append("KIF（簡易表示）")
        lines.append(f"手番: {getattr(self.pos, 'side_to_move', '?')}")
        moves = getattr(self.pos, "moves", []) or []
        lines.append(f"手数: {len(moves)}")
        lines.append("")
        for i, mv in enumerate(moves, start=1):
            try:
                if isinstance(mv, dict):
                    kind = mv.get("kind")
                    frm = mv.get("from")
                    to = mv.get("to")
                    promote = bool(mv.get("promote"))
                    drop = bool(mv.get("drop"))
                else:
                    kind = getattr(mv, "kind", None)
                    frm = getattr(mv, "frm", getattr(mv, "from_sq", getattr(mv, "from_", None)))
                    to = getattr(mv, "to", getattr(mv, "to_sq", None))
                    promote = bool(getattr(mv, "promote", False))
                    drop = bool(getattr(mv, "drop", False))
                lines.append(kif_line_for_minimal_move(i, kind, frm, to, promote, drop))
            except Exception as e:
                lines.append(f"{i} ???  # 変換エラー: {e}")
        return "\n".join(lines)

    # ---- アクション ----

    def action_clear_square(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        sq = (self.cursor.file, self.cursor.rank)
        self._push_history()
        self.pos.set_piece(sq, None)
        self._refresh()
        self._set_status()
        if self.log_widget:
            self.log_widget.write(f"[配置] {_square_label(*sq)} <- 消去")

    def _open_piece_picker(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.push_screen(PiecePicker(self.last_piece_token), callback=self._on_piece_picked)

    def action_open_piece_picker(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self._open_piece_picker()

    def _on_piece_picked(self, tok: Optional[str]) -> None:
        if not tok:
            return
        tok = tok.strip()
        if not tok:
            return
        try:
            p = parse_piece_token(tok)
            sq = self._pending_square or (self.cursor.file, self.cursor.rank)
            self._pending_square = None

            self._push_history()
            self.pos.set_piece(sq, p)

            if tok != ".":
                self.last_piece_token = tok

            self._refresh()
            self._set_status()
            if self.log_widget:
                placed = "消去" if p is None else tok
                self.log_widget.write(f"[配置] {_square_label(*sq)} <- {placed}")
        except Exception as e:
            if self.log_widget:
                self.log_widget.write(f"[エラー] {e}")

    def action_open_hand_picker(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.push_screen(HandPicker(default_color="B"), callback=self._on_hand_picked)

    def _on_hand_picked(self, result: Optional[tuple[str, str, int]]) -> None:
        if not result:
            return
        color, kind, n = result
        try:
            self._push_history()
            self.pos.set_hand(color, kind, n)
            self._refresh()
            self._set_status()
            if self.log_widget:
                side = "先手" if color == "B" else "後手"
                self.log_widget.write(f"[持駒] {side} {PIECE_JP.get(kind, kind)}={n}")
        except Exception as e:
            if self.log_widget:
                self.log_widget.write(f"[エラー] {e}")

    # ---- キー処理 ----

    async def on_key(self, event: Key) -> None:
        if self.mode == Mode.NORMAL and event.key == "q":
            await self.action_quit()
            return

        if self.mode == Mode.NORMAL:
            if event.key in ("h", "left"):
                self.cursor.file = max(1, min(9, self.cursor.file + 1))
                self._refresh()
                self._set_status()
                event.stop()
                return
            if event.key in ("l", "right"):
                self.cursor.file = max(1, min(9, self.cursor.file - 1))
                self._refresh()
                self._set_status()
                event.stop()
                return
            if event.key in ("k", "up"):
                self.cursor.rank = max(1, min(9, self.cursor.rank - 1))
                self._refresh()
                self._set_status()
                event.stop()
                return
            if event.key in ("j", "down"):
                self.cursor.rank = max(1, min(9, self.cursor.rank + 1))
                self._refresh()
                self._set_status()
                event.stop()
                return

            if event.key == "enter":
                # ★Enterの“余波”がピッカーへ回り込むのを避けるため、call_laterで開く
                self._pending_square = (self.cursor.file, self.cursor.rank)
                event.prevent_default()
                event.stop()
                self.call_later(self._open_piece_picker)
                return

        # INPUTモードのキー処理は Input が担当

    # ---- INPUTコマンド ----

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.mode != Mode.INPUT:
            return

        line = event.value.strip()
        if self.cmd:
            self.cmd.value = ""

        if not line:
            return

        if self.log_widget:
            self.log_widget.write(f"> {line}")

        # :q
        if line in (":q", ":quit"):
            await super().action_quit()
            return

        # 数字入力：7776 / 77761 / 076（打ち）
        if re.fullmatch(r"\d{3,5}", line):
            try:
                tag, kind, frm, to, promote = self.pos.parse_numeric(line)

                # 変更前に履歴を積む（undoを確実に）
                self._push_history()

                if tag == "drop_pick":
                    cands = self.pos.drop_candidates(to)
                    if not cands:
                        raise ValueError("そのマスに打てる持ち駒がありません（または既に駒があります）")
                    if len(cands) == 1:
                        self.pos.apply_move_minimal(cands[0], None, to, False, True)
                        if self.log_widget:
                            self.log_widget.write(f"[指し手] 打ち {cands[0]} -> {to}")
                        self._refresh()
                        self._set_status()
                        return

                    picked = await self.push_screen_wait(DropPicker(cands, to))
                    if picked is None:
                        if self.log_widget:
                            self.log_widget.write("[案内] 打ちはキャンセルしました")
                        return

                    self.pos.apply_move_minimal(picked, None, to, False, True)
                    if self.log_widget:
                        self.log_widget.write(f"[指し手] 打ち {picked} -> {to}")
                    self._refresh()
                    self._set_status()
                    return

                # 通常の移動
                self.pos.apply_move_minimal(kind, frm, to, promote, False)
                if self.log_widget:
                    prom = "（成り）" if promote else ""
                    self.log_widget.write(f"[指し手] {frm} -> {to} {prom}".rstrip())
                self._refresh()
                self._set_status()
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        # コマンド解析（大文字小文字OK）
        parts = line.split()
        cmd = parts[0].lower()
        arg = " ".join(parts[1:]) if len(parts) > 1 else ""

        if cmd in ("help", "?"):
            if self.log_widget:
                self.log_widget.write(HelpView().render())
            return

        if cmd == "show":
            if self.log_widget:
                self.log_widget.write(
                    f"[状態] 手番={getattr(self.pos, 'side_to_move', '?')}  手数={len(getattr(self.pos, 'moves', []) or [])}"
                )
            return

        # p <sq> <token>：駒配置（P でも p でもOK）
        if cmd == "p":
            try:
                m = re.fullmatch(r"(\d{2})\s+(\S+)", arg.strip())
                if not m:
                    raise ValueError("使い方: p <sq> <token>（例: p 55 P / p 33 vK / p 11 .）")
                sq2 = m.group(1)
                tok = m.group(2)

                f = int(sq2[0])
                r = int(sq2[1])
                if not (1 <= f <= 9 and 1 <= r <= 9):
                    raise ValueError("sq は 11..99 の範囲で指定してください")

                self._push_history()
                p = parse_piece_token(tok)
                self.pos.set_piece((f, r), p)
                self._refresh()
                self._set_status()
                if self.log_widget:
                    self.log_widget.write(f"[配置] {sq2} <- {('消去' if tok=='.' else tok)}")
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        # h <b|w> <KIND> <N>：持駒設定
        if cmd == "h":
            try:
                ps = arg.split()
                if len(ps) != 3:
                    raise ValueError("使い方: h <b|w> <KIND> <N>（例: h b G 1）")
                side = ps[0].lower()
                kind = ps[1].upper()
                n = int(ps[2])

                color = "B" if side == "b" else "W"
                self._push_history()
                self.pos.set_hand(color, kind, n)
                self._refresh()
                self._set_status()
                if self.log_widget:
                    jp = "先手" if color == "B" else "後手"
                    self.log_widget.write(f"[持駒] {jp} {PIECE_JP.get(kind, kind)}={n}")
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if cmd == "start":
            try:
                self._push_history()
                startpos = "lnsgkgsnl/1r5b1/p1ppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b - 1"
                snap = sfen_to_snapshot(startpos)

                # ★3要素/4要素どちらでもOKにする
                if isinstance(snap, (list, tuple)) and len(snap) == 3:
                    board, hands, stm = snap
                    self.pos.board, self.pos.hands, self.pos.side_to_move = board, hands, stm
                    self.pos.moves = []
                else:
                    self.pos.board, self.pos.hands, self.pos.side_to_move, self.pos.moves = snap

                self.pos._history = []
                self._refresh()
                self._set_status()
                if self.log_widget:
                    self.log_widget.write("[OK] 初期局面を読み込みました")
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if cmd == "clear":
            self._push_history()
            self.pos.clear_all()
            if self.log_widget:
                self.log_widget.write("[OK] 盤面・持駒・手順をクリアしました")
            self._refresh()
            self._set_status()
            return

        if cmd == "undo":
            try:
                self.pos.undo()
                if self.log_widget:
                    self.log_widget.write("[OK] 1手戻しました")
                self._refresh()
                self._set_status()
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if cmd == "sfen" and not arg:
            try:
                hands_b = _get_hand_dict(self.pos, "B")
                hands_w = _get_hand_dict(self.pos, "W")
                s = snapshot_to_sfen(self.pos.board, hands_b, self.pos.side_to_move, hands_w)
                if self.log_widget:
                    self.log_widget.write("[SFEN] " + s)
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if cmd in ("load", "sfen") and arg:
            try:
                self._push_history()
                snap = sfen_to_snapshot(arg)

                if isinstance(snap, (list, tuple)) and len(snap) == 3:
                    board, hands, stm = snap
                    self.pos.board, self.pos.hands, self.pos.side_to_move = board, hands, stm
                    self.pos.moves = []
                else:
                    self.pos.board, self.pos.hands, self.pos.side_to_move, self.pos.moves = snap

                self.pos._history = []
                if self.log_widget:
                    self.log_widget.write("[OK] SFENを読み込みました")
                self._refresh()
                self._set_status()
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if cmd == "kif":
            try:
                text = self._generate_kif_text()
                await self.push_screen(KifViewer(text))
            except Exception as e:
                if self.log_widget:
                    self.log_widget.write(f"[エラー] {e}")
            return

        if self.log_widget:
            self.log_widget.write("[案内] 不明なコマンドです。help で一覧、または数字入力（例: 7776）を試してください。")


if __name__ == "__main__":
    ShogiTui().run()
