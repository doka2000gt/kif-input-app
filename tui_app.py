#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Static, Input, RichLog

from constants import PIECE_JP, KIND_TO_PYO
from position import ShogiPosition
from models import Piece
from sfen import sfen_to_snapshot, snapshot_to_sfen, compute_gote_remaining
from manual_kif import kif_line_for_minimal_move, board_map_to_piyo
from helpers import now_yyyy_mm_dd_hhmmss, inv_count_kanji, _write_kif_unique
from paths import _ensure_output_dir


# ----------------- helpers -----------------

Square = Tuple[int, int]


def _cell_str(pos: ShogiPosition, f: int, r: int) -> str:
    """盤面セルの表示（2文字幅想定：' ・' / ' 歩' / 'v歩' / 'v竜' など）"""
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
    """盤面表示（カーソル位置は reverse でハイライト）"""

    def __init__(self, tui: "ShogiTui"):
        super().__init__()
        self.tui = tui

    def render(self) -> Text:
        pos = self.tui.pos
        cur = self.tui.cursor

        # 表示幅が崩れる最大要因：成香/成桂/成銀（全角2字）
        # 盤面表示だけ「杏/圭/全」で1字にして、セル幅を揃える
        DISP_NAME = {
            ("P", False): "歩", ("L", False): "香", ("N", False): "桂", ("S", False): "銀",
            ("G", False): "金", ("B", False): "角", ("R", False): "飛", ("K", False): "玉",
            ("P", True): "と", ("L", True): "杏", ("N", True): "圭", ("S", True): "全",
            ("B", True): "馬", ("R", True): "竜",
        }

        # 全角数字（列ラベルに使う）
        FW = {1: "１", 2: "２", 3: "３", 4: "４", 5: "５", 6: "６", 7: "７", 8: "８", 9: "９"}

        def cell_str(f: int, r: int) -> str:
            p = pos.board.get((f, r))
            if p is None:
                return " ・"
            name = DISP_NAME.get((p.kind, bool(p.prom)), PIECE_JP[p.kind])
            return ("v" + name) if p.color == "W" else (" " + name)

        t = Text()

        # 上の手駒表示
        t.append(f"▽持駒: {_format_hand(pos, 'W')}\n")

        # 列ラベル（9→1）
        t.append("    ")  # 行ラベル分の余白
        for f in range(9, 0, -1):
            t.append(f" {FW[f]}")
        t.append("\n")

        # 盤面（行ラベル付き：1→9）
        for r in range(1, 10):
            t.append(f" {r}  ")
            for f in range(9, 0, -1):
                cell = cell_str(f, r)
                if (f, r) == (cur.file, cur.rank):
                    t.append_text(Text(cell, style="reverse"))
                else:
                    t.append(cell)
            t.append("\n")

        # 下の列ラベル（もう一回）
        t.append("    ")
        for f in range(9, 0, -1):
            t.append(f" {FW[f]}")
        t.append("\n")

        # 下の手駒表示
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
            event.stop()
            self.dismiss(None)


# （HandPicker / DropPicker / PlacementMode は、あなたの添付版のまま）
# ここから下もそのまま貼っています（省略せず全体を残します）

class HandPicker(ModalScreen[Optional[list[Tuple[str, str, int]]]]):
    MAX_HAND = 18

    def __init__(self, in_setup: bool):
        super().__init__()
        self.in_setup = in_setup

    def compose(self) -> ComposeResult:
        help_txt = (
            "持駒ピッカー（複数入力OK）\n"
            "  1項目: [b|w]<駒><枚数>  または  <駒><枚数>\n"
            "  ※スペースは省略可（例: P1 / bP1 / wR2）\n"
            "  区切り: 改行 / ; / ,\n"
            f"  枚数: 1〜{self.MAX_HAND}（0は禁止）\n"
            "  駒: P L N S G B R K\n"
            "  Enter: 決定 / Esc(q): キャンセル\n"
        )
        yield Vertical(
            Static(help_txt),
            Input(placeholder="例: P1; G1; R2", id="hand"),
        )

    def on_mount(self) -> None:
        self.query_one("#hand", Input).focus()

    async def on_key(self, event: Key) -> None:
        if event.key in ("escape", "q"):
            event.stop()
            self.dismiss(None)

    def _validate_n(self, n: int) -> None:
        if n <= 0:
            raise ValueError("枚数 0 は指定できません（1以上にしてください）")
        if n > self.MAX_HAND:
            raise ValueError(f"枚数が大きすぎます（上限 {self.MAX_HAND}）")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()

        raw = event.value.strip()
        if not raw:
            return

        raw = raw.replace(",", "\n").replace(";", "\n")

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        parsed: list[Tuple[str, str, int]] = []

        try:
            for ln in lines:
                m3 = re.fullmatch(r"(b|w)\s*([PLNSGBRK])\s*(\d+)", ln, re.IGNORECASE)
                m2 = re.fullmatch(r"([PLNSGBRK])\s*(\d+)", ln, re.IGNORECASE)

                if m3:
                    side = m3.group(1).lower()
                    kind = m3.group(2).upper()
                    n = int(m3.group(3))
                    self._validate_n(n)
                    color = "B" if side == "b" else "W"
                    parsed.append((color, kind, n))
                    continue

                if m2:
                    kind = m2.group(1).upper()
                    n = int(m2.group(2))
                    self._validate_n(n)
                    if self.in_setup:
                        parsed.append(("B", kind, n))
                    else:
                        raise ValueError("対局モードでは b/w を付けてください（例: bP1）")
                    continue

                raise ValueError(f"形式が不正です: '{ln}'（例: P1 / bP1 / wR2）")

        except Exception as e:
            self.query_one("#hand", Input).value = raw
            try:
                app = self.app  # type: ignore
                if hasattr(app, "log_err"):
                    app.log_err(f"持駒入力エラー: {e}")
            except Exception:
                pass
            return

        self.dismiss(parsed)


class DropPicker(ModalScreen[Optional[str]]):
    selected: int = reactive(0)

    def __init__(self, to_sq: Square, candidates: list[str]):
        super().__init__()
        self.to_sq = to_sq
        order = ["R", "B", "G", "S", "N", "L", "P", "K"]
        self.candidates = [k for k in order if k in candidates] + [k for k in candidates if k not in order]
        if not self.candidates:
            self.candidates = candidates[:]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("", id="drop_menu"),
            Static("j/k(↑/↓):選択  Enter:決定  Esc(q):キャンセル  直接P等でも可", id="drop_help"),
        )

    def on_mount(self) -> None:
        self.set_focus(self)
        self._refresh_menu()

    def watch_selected(self, _: int) -> None:
        self._refresh_menu()

    def _refresh_menu(self) -> None:
        menu = Text()
        menu.append("打ち候補が複数あります。どの駒を打ちますか？\n")
        menu.append(f"  行先: {self.to_sq}\n\n")
        for i, k in enumerate(self.candidates):
            mark = "▶" if i == self.selected else " "
            menu.append(f"{mark} {k} : {PIECE_JP[k]}\n")
        self.query_one("#drop_menu", Static).update(menu)

    async def on_key(self, event: Key) -> None:
        k = event.key

        if k in ("escape", "q"):
            event.stop()
            self.dismiss(None)
            return

        if k in ("j", "down"):
            event.stop()
            if self.candidates:
                self.selected = (self.selected + 1) % len(self.candidates)
            return

        if k in ("k", "up"):
            event.stop()
            if self.candidates:
                self.selected = (self.selected - 1) % len(self.candidates)
            return

        if k == "enter":
            event.stop()
            if self.candidates:
                self.dismiss(self.candidates[self.selected])
            else:
                self.dismiss(None)
            return

        if len(k) == 1 and re.fullmatch(r"[plnsgbrkPLNSGBRK]", k):
            event.stop()
            tok = k.upper()
            if tok in self.candidates:
                self.dismiss(tok)
            return


class PlacementMode(ModalScreen[None]):
    """
    連続配置モード（盤面編集用・盤面も表示する版）

    操作:
      - h/j/k/l または ←↓↑→ で盤面カーソル移動
      - P/L/N/S/G/B/R/K で駒配置（※大文字推奨：h/l と衝突回避）
      - v で先手/後手トグル
      - + で成りトグル（P/L/N/S/B/Rに有効）
      - . または Backspace/Delete で消去
      - Esc / q で終了
    """

    selected: int = reactive(0)

    def __init__(self, in_setup: bool):
        super().__init__()
        self.in_setup = in_setup
        self.color = "B"       # 先手/後手
        self.prom_next = False # 次の1回だけ成り
        self.last_kind: Optional[str] = None  # 直近の駒（拾う/置く）
        self.panel_open = False
        self.panel_index = 0
        self.panel_items = ["P", "L", "N", "S", "G", "B", "R", "K"]

        # 画面内に盤面を持たせる（本体の board_view は使い回せないため別インスタンス）
        self._board: Optional[BoardView] = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Static("連続配置モード", id="place_title"),
                # 盤面（リアルタイム表示）
                BoardView(self.app),  # type: ignore[arg-type]
                id="place_left",
            ),
            Vertical(
                Static("", id="place_help"),
                Static("", id="piece_panel"),
                id="place_right",
            ),
            id="place_root",
        )

    def on_mount(self) -> None:
        self.set_focus(self)
        # BoardView を取得して保持
        self._board = self.query_one(BoardView)
        self._refresh_help()
        self._refresh_panel()
        if self._board:
            self._board.refresh()

    def _refresh_help(self) -> None:
        app: ShogiTui = self.app  # type: ignore
        f, r = app.cursor.file, app.cursor.rank
        side = "先手" if self.color == "B" else "後手"
        prom = "成" if self.prom_next else "不成"
        mode = "編集(先手固定)" if app.in_setup else "対局(手番交互)"

        # 現在マスの駒情報
        cur_piece = app.pos.board.get((f, r))
        if cur_piece is None:
            cur_txt = "空"
        else:
            cur_txt = f"{'後手' if cur_piece.color=='W' else '先手'}{'成' if cur_piece.prom else ''}{PIECE_JP[cur_piece.kind]}"
        picked = (f"{PIECE_JP[self.last_kind]}" if self.last_kind else "なし")

        t = Text()
        t.append(f"連続配置モード  [{mode}]\n")
        t.append(f"  位置: ({f},{r})  /  現在: {cur_txt}  /  直近: {picked}\n")
        t.append(f"  配置側: {side}  /  次の成り: {prom}\n\n")
        t.append("  移動: h/j/k/l または ←↓↑→\n")
        t.append("  置く: P L N S G B R K（大文字推奨）\n")
        t.append("  先後: v でトグル / 成り: + でトグル\n")
        t.append("  消去: . / Backspace / Delete\n")
        t.append("  終了: Esc / q\n")
        self.query_one("#place_help", Static).update(t)

    def _refresh_panel(self) -> None:
        panel = Text()
        if not self.panel_open:
            panel.append("Tab: 駒パネルを開く\n")
            self.query_one("#piece_panel", Static).update(panel)
            return

        panel.append("駒パネル（j/k, ↑/↓ で選択 / Enterで配置 / Tab・Escで閉じる）\n\n")
        for i, k in enumerate(self.panel_items):
            mark = "▶" if i == self.panel_index else " "
            panel.append(f"{mark} {k} : {PIECE_JP[k]}\n")

        self.query_one("#piece_panel", Static).update(panel)

    def _refresh_all(self, sync: bool = True) -> None:
        # sync=True のときだけ同期（空マスでv/+を押した直後は保持したい）
        if sync:
            self._sync_from_square()

        if self._board:
            self._board.refresh()
        self._refresh_help()
        self._refresh_panel()

    def _place(self, kind: str) -> None:
        app: ShogiTui = self.app  # type: ignore
        f, r = app.cursor.file, app.cursor.rank

        can_promote = kind in ("P", "L", "N", "S", "B", "R")
        prom = self.prom_next and can_promote

        app.pos.board[(f, r)] = Piece(color=self.color, kind=kind, prom=prom)

        # 置いたら毎回リセット
        self.last_kind = kind
        self.color = "B"
        self.prom_next = False

        # 編集モード中は手番を先手固定（既存方針）
        if hasattr(app, "_force_setup_side"):
            app._force_setup_side()

        app.log_ok(f"配置: {(f, r)} <- {'v' if self.color=='W' else ''}{'+' if prom else ''}{kind}")
        self._refresh_all(sync=False)

    def _clear_square(self) -> None:
        app: ShogiTui = self.app  # type: ignore
        f, r = app.cursor.file, app.cursor.rank
        app.pos.board.pop((f, r), None)

        if hasattr(app, "_force_setup_side"):
            app._force_setup_side()

        app.log_ok(f"消去: {(f, r)}")
        self._refresh_all()

    def _sync_from_square(self) -> None:
        """カーソル位置の駒に合わせて編集状態を同期する（自動拾い）"""
        app: ShogiTui = self.app  # type: ignore
        f, r = app.cursor.file, app.cursor.rank
        p = app.pos.board.get((f, r))

        if p is None:
            # 空マスならリセット値
            self.color = "B"
            self.prom_next = False
            self.last_kind = None
        else:
            self.color = p.color
            self.prom_next = bool(p.prom)
            self.last_kind = p.kind

    async def on_key(self, event: Key) -> None:
        k = event.key
        app: ShogiTui = self.app  # type: ignore

        # 終了
        if k in ("escape", "q"):
            event.stop()
            app.board_view.refresh()
            self.dismiss(None)
            return

        # --- 駒パネルの開閉 ---
        if k == "tab":
            event.stop()
            self.panel_open = not self.panel_open
            if self.panel_open:
                self.panel_index = 0
            self._refresh_all(sync=False)
            return

        # --- パネルが開いている間は、j/k(↑/↓)/Enter をパネル操作に割り当て ---
        if self.panel_open:
            if k in ("escape", "q"):
                event.stop()
                self.panel_open = False
                self._refresh_all(sync=False)
                return

            if k in ("j", "down"):
                event.stop()
                self.panel_index = (self.panel_index + 1) % len(self.panel_items)
                self._refresh_all(sync=False)
                return

            if k in ("k", "up"):
                event.stop()
                self.panel_index = (self.panel_index - 1) % len(self.panel_items)
                self._refresh_all(sync=False)
                return

            if k == "enter":
                event.stop()
                kind = self.panel_items[self.panel_index]
                self._place(kind)
                self.panel_open = False
                self._refresh_all(sync=False)
                return

        # 移動
        if k in ("h", "left"):
            event.stop()
            app.action_left()
            self._refresh_all()
            return
        if k in ("j", "down"):
            event.stop()
            app.action_down()
            self._refresh_all()
            return
        if k in ("k", "up"):
            event.stop()
            app.action_up()
            self._refresh_all()
            return
        if k in ("l", "right"):
            event.stop()
            app.action_right()
            self._refresh_all()
            return

        # 消去
        if k in (".", "backspace", "delete"):
            event.stop()
            self._clear_square()
            return

        # 先後トグル
        if k == "v":
            event.stop()
            f, r = app.cursor.file, app.cursor.rank
            p = app.pos.board.get((f, r))
            if p is not None:
                p.color = "W" if p.color == "B" else "B"
                app.pos.board[(f, r)] = p
                app.log_ok(f"編集: {(f, r)} の先後を反転しました")
                if hasattr(app, "_force_setup_side"):
                    app._force_setup_side()
                self._refresh_all()
            else:
                self.color = "W" if self.color == "B" else "B"
                self._refresh_all(sync=False)
            return

        # 成りトグル
        if k in ("+", "plus"):
            event.stop()
            f, r = app.cursor.file, app.cursor.rank
            p = app.pos.board.get((f, r))
            if p is not None:
                can_promote = p.kind in ("P", "L", "N", "S", "B", "R")
                if not can_promote:
                    app.log_err("この駒は成れません")
                    self._refresh_all()
                    return
                p.prom = not p.prom
                app.pos.board[(f, r)] = p
                app.log_ok(f"編集: {(f, r)} の成/不成を反転しました")
                if hasattr(app, "_force_setup_side"):
                    app._force_setup_side()
                self._refresh_all()
            else:
                self.prom_next = not self.prom_next
                self._refresh_all(sync=False)
            return

        # 駒配置：大文字のみ（h/l衝突回避）
        if len(k) == 1 and re.fullmatch(r"[PLNSGBRK]", k):
            event.stop()
            kind = k.upper()
            self._place(kind)
            return


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

        Binding("H", "open_hand_picker", "hand", show=False),
        Binding("P", "open_place_mode", "place", show=False),
    ]

    mode = reactive(Mode.NORMAL)

    def __init__(self) -> None:
        super().__init__()
        self.pos = ShogiPosition()
        self.cursor = Cursor()
        self.in_setup = True  # ★編集モード（startまで先手固定）

        # ★KIF出力用：開始局面スナップショット（start時に確定）
        self.start_snapshot = self.pos.clone_state()

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
        self.log_ok("起動しました（編集モード：先手固定）")

    def _set_title(self) -> None:
        suffix = "編集(先手固定)" if self.in_setup else "対局(手番交互)"
        self.title = f"将棋KIF入力TUI  [{self.mode.value}]  [{suffix}]"

    def _force_setup_side(self) -> None:
        """編集モード中は常に先手番に戻す"""
        if self.in_setup:
            self.pos.side_to_move = "B"

    # --- logging helpers ---
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

    # --- save helpers ---
    def _normalize_kif_filename(self, name: str) -> str:
        name = name.strip()
        if not name:
            return name
        if name.lower().endswith(".kif"):
            return name
        return name + ".kif"

    def _save_kif(self, filename: Optional[str] = None) -> None:
        import datetime as _dt

        text = self._generate_kif_text()

        if filename and filename.strip():
            filename = self._normalize_kif_filename(filename)
        else:
            filename = "export_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".kif"

        outdir = _ensure_output_dir()
        saved = _write_kif_unique(outdir, filename, text, seen=None)
        if saved is None:
            self.log_ok("同一内容のため保存をスキップしました")
        else:
            self.log_ok(f"保存しました: {saved}")

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

    def action_open_place_mode(self) -> None:
        try:
            # あなたの元ファイルでは PlacementMode が存在します（ここはそのまま）
            self.push_screen(PlacementMode(self.in_setup))  # type: ignore[name-defined]
            self.log_state("連続配置モードを開始しました")
        except Exception as e:
            self.log_err(f"連続配置モード起動失敗: {e}")

    # --- cursor move actions ---
    def action_left(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.file = min(9, self.cursor.file + 1)
        self.board_view.refresh()

    def action_right(self) -> None:
        if self.mode != Mode.NORMAL:
            return
        self.cursor.file = max(1, self.cursor.file - 1)
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

    # --- hand picker (UI) ---
    def action_open_hand_picker(self) -> None:
        try:
            self.push_screen(HandPicker(self.in_setup), self._on_hand_picker_done)
        except Exception as e:
            self.log_err(f"持駒ピッカー失敗: {e}")

    def _on_hand_picker_done(self, res) -> None:
        try:
            if res is None:
                return

            if not hasattr(self.pos, "hands") or not isinstance(self.pos.hands, dict):
                raise ValueError("pos.hands が見つかりません")

            for (color, kind, n) in res:
                if color not in self.pos.hands:
                    self.pos.hands[color] = {}
                self.pos.hands[color][kind] = int(n)

            self.log_ok(f"持駒設定(UI): {len(res)}件")
            self._force_setup_side()
            self._set_title()
            self.board_view.refresh()

        except Exception as e:
            self.log_err(f"持駒ピッカー後処理失敗: {e}")

    def _on_drop_picker_done(self, ctx, res) -> None:
        try:
            if res is None:
                self.log_err("打ちをキャンセルしました")
                return

            to_sq = ctx["to"]
            kind = res
            self.pos.apply_move_minimal(kind, None, to_sq, False, True)

            self._force_setup_side()

            self.log_ok(f"drop {kind} -> {to_sq}")
            self._set_title()
            self.board_view.refresh()

        except Exception as e:
            self.log_err(f"打ちの適用に失敗: {e}")

    # ---- KIF (完成版) ----
    def _generate_kif_text(self) -> str:
        """
        仕様：
          - 盤面図は「開始局面」（start時に確定したスナップショット）を出す
          - 最終行は「までN手で詰み」（消費時間なし）
          - 指し手の消費時間は 1手=1秒、通算も+1秒ずつ
        """
        out: list[str] = []

        # --- ヘッダ（寄せたいフォーマット） ---
        out.append("# ----  ANKIF向け / 自作詰将棋メーカー by TUI  ----")
        out.append("手合割：詰将棋")
        out.append("先手：先手")
        out.append("後手：後手")

        # --- 開始局面スナップショット ---
        board0, hands0, _side0, _moves0 = self.start_snapshot
        hands0_b = hands0.get("B", {})

        # 後手の持駒（開始局面ベース）
        gote_rem = compute_gote_remaining(board0, hands0_b)

        def _hands_dict_to_piyo(d: Dict[str, int]) -> str:
            order = ["R", "B", "G", "S", "N", "L", "P"]
            parts: list[str] = []
            for k in order:
                n = int(d.get(k, 0) or 0)
                if n <= 0:
                    continue
                # 見本の全角スペースに寄せる（気持ち）
                parts.append(PIECE_JP[k] + inv_count_kanji(n))
            return "　".join(parts) + ("　" if parts else "")

        out.append("後手の持駒：" + _hands_dict_to_piyo(gote_rem))
        out.append(board_map_to_piyo(board0))
        out.append("先手の持駒：" + _hands_dict_to_piyo(hands0_b))

        # 日時は見本どおり「盤面の後ろ」
        out.append(f"終了日時：{now_yyyy_mm_dd_hhmmss()}")

        out.append("手数----指手---------消費時間--")

        prev_to = None
        total_sec = 0
        sec_per_move = 1

        for idx, mv in enumerate(self.pos.moves, start=1):
            try:
                total_sec += sec_per_move
                line, prev_to = kif_line_for_minimal_move(
                    idx, mv, prev_to, sec_per_move, total_sec
                )
            except Exception:
                line = f"{idx} {mv}"
                prev_to = getattr(mv, "to_sq", None)

            # 指手と消費時間の間は半角スペース1つ
            line = re.sub(r"\s+\(", " (", line)
            # "(" の直後の余分なスペースを削除
            line = re.sub(r"\(\s+", "(", line)

            out.append(line)

        # 終局行（あなたの希望：消費時間なし）
        if self.pos.moves:
            out.append(f"まで{len(self.pos.moves)}手で詰み")

        return "\n".join(out) + "\n"

    # ---- Command input ----
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""

        if not line:
            return

        self._log(f"> {line}")

        # --- numeric move input ---
        if re.fullmatch(r"\d{3,5}", line):
            try:
                tag, kind, frm, to, promote = self.pos.parse_numeric(line)

                # ---- drop ----
                if tag == "drop_pick":
                    cands = self.pos.drop_candidates(to)
                    if not cands:
                        raise ValueError("打てる持ち駒がありません（またはそのマスに駒があり ます）")

                    if len(cands) == 1:
                        self.pos.apply_move_minimal(cands[0], None, to, False, True)
                        self._force_setup_side()
                        self.log_ok(f"drop {cands[0]} -> {to}")
                        self._set_title()
                        self.board_view.refresh()
                        return

                    if os.environ.get("SMOKE") == "1":
                        raise ValueError("複数候補の打ち分けは未対応（SMOKE中）")

                    ctx = {"to": to, "cands": cands}
                    self.push_screen(DropPicker(to, cands), lambda res: self._on_drop_picker_done(ctx, res))
                    self.log_state(f"打ち候補選択: to={to}, cands={cands}")
                    return

                # ---- normal move ----
                p = self.pos.board.get(frm)
                if p is None:
                    raise ValueError("移動元に駒がありません")
                self.pos.apply_move_minimal(p.kind, frm, to, promote, False)

                self._force_setup_side()

                self.log_ok(f"move {frm}->{to}{' promote' if promote else ''}")
                self._set_title()
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
            self.log_err("INPUT中の q は終了しません。:q を入力してください")
            return

        if cmd in ("help", "?"):
            self._log("[HINT] Commands: show | start | sfen | load <SFEN> | clear/new/reset | undo | kif | kif export <name> | s [name] | h <b|w> <KIND> <N> | help/? | :q")
            self._log("[HINT] Moves: 7776 (move) / 77761 (promote) / 076 (drop: 0+file+rank)")
            self._log("[HINT] Hand UI: press 'H'")
            self._log("[HINT] Setup mode: startまで先手固定（連続で配置できます）")
            return

        if cmd == "show":
            self.log_state(f"手番={self.pos.side_to_move}, 手数={len(self.pos.moves)}")
            return

        # ---- save short command ----
        if cmd == "s":
            try:
                name = arg.strip() if arg else ""
                self._save_kif(name if name else None)
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "start":
            try:
                # ★開始局面を確定（KIFの盤面図に使う）
                self.start_snapshot = self.pos.clone_state()

                # 盤面/持駒はそのままに、編集→対局へ
                self.in_setup = False
                self.pos.clear_moves()  # moves と history をクリア

                self._set_title()
                self.log_ok("開始局面を確定しました（対局モード：手番交互）")
                self.board_view.refresh()
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd in ("clear", "new", "reset"):
            self.pos.clear_all()
            self.start_snapshot = self.pos.clone_state()
            self.in_setup = True
            self._force_setup_side()
            self._set_title()
            self.log_ok("盤面・持駒・手順をクリアしました（編集モード：先手固定）")
            self.cursor.file = 5
            self.cursor.rank = 5
            self.board_view.refresh()
            return

        if cmd == "undo":
            try:
                self.pos.undo()
                self._force_setup_side()
                self.log_ok("undo")
                self._set_title()
                self.board_view.refresh()
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "sfen" and not arg:
            try:
                board, hands, side_to_move, _moves = self.pos.clone_state()
                hands_b = hands.get("B", {})
                gote_auto = compute_gote_remaining(board, hands_b)
                s = snapshot_to_sfen(board, hands_b, side_to_move, gote_auto)
                self.log_sfen(s)
            except Exception as e:
                self.log_err(str(e))
            return

        if cmd in ("load", "sfen") and arg:
            try:
                board_map, hands_b, side_to_move = sfen_to_snapshot(arg)

                self.pos.board = board_map
                self.pos.hands = {"B": hands_b, "W": {}}
                self.pos.side_to_move = side_to_move
                self.pos.moves = []
                self.pos._history = []

                # SFEN読込は「局面編集」とみなす（startで対局へ）
                self.in_setup = True
                self._force_setup_side()
                self._set_title()
                self.log_ok("SFENを読み込みました（編集モード：先手固定）")
                self.board_view.refresh()

                # ★読み込んだ局面を「開始局面候補」にもしておく
                self.start_snapshot = self.pos.clone_state()

            except Exception as e:
                self.log_err(str(e))
            return

        # 持駒設定（テキストコマンド）
        if cmd == "h":
            try:
                m = re.fullmatch(r"(b|w)\s+([PLNSGBRK])\s+(\d+)", arg.strip(), re.IGNORECASE)
                if not m:
                    raise ValueError("使い方: h <b|w> <KIND> <N>   例: h b P 1 / h w R 2")
                side = m.group(1).lower()
                kind = m.group(2).upper()
                n = int(m.group(3))
                color = "B" if side == "b" else "W"

                if not hasattr(self.pos, "hands") or not isinstance(self.pos.hands, dict):
                    raise ValueError("pos.hands が見つかりません")

                if color not in self.pos.hands:
                    self.pos.hands[color] = {}
                self.pos.hands[color][kind] = n

                self.log_ok(f"持駒設定: {color} {kind} {n}")
                self._force_setup_side()
                self._set_title()
                self.board_view.refresh()

            except Exception as e:
                self.log_err(str(e))
            return

        if cmd == "kif":
            try:
                a = arg.strip()
                if not a:
                    text = self._generate_kif_text()
                    self.log_kif("出力しました")
                    await self.push_screen(KifViewer(text))
                    return

                toks = a.split()
                sub = toks[0]
                if sub in ("export", "save"):
                    text = self._generate_kif_text()
                    if len(toks) >= 2:
                        filename = self._normalize_kif_filename(toks[1])
                    else:
                        import datetime as _dt
                        filename = "export_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".kif"

                    outdir = _ensure_output_dir()
                    saved = _write_kif_unique(outdir, filename, text, seen=None)
                    if saved is None:
                        self.log_ok("同一内容のため保存をスキップしました")
                    else:
                        self.log_ok(f"保存しました: {saved}")
                    return

                text = self._generate_kif_text()
                self.log_kif("出力しました")
                await self.push_screen(KifViewer(text))
            except Exception as e:
                self.log_err(str(e))
            return

        self.log_err("unknown command. type 'help' or enter numeric move.")


if __name__ == "__main__":
    ShogiTui().run()
