#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Textualアプリ用 Smoke Runner（シナリオ再生＋ログ/JSONL出力）

使い方（例）:
  PYTHONPATH=. python -m tests.smoke_runner tests/smoke_scenario.txt --out smoke.log --jsonl smoke.jsonl

Makefileから:
  set PYTHONPATH=. && python -m tests.smoke_runner tests/smoke_scenario.txt --out smoke.log --jsonl smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Any

# ★プロジェクト構成に合わせて調整（通常これでOK）
from tui_app import ShogiTui


# -----------------------------
# Scenario DSL
# -----------------------------
# 1行1ステップ（先頭/末尾の空白は無視）
#   WAIT <ms>
#   KEY <key>
#   CMD <text>          # cmd入力欄に入れてEnter
#   EXPECT <text>       # RichLog内に含まれること
#   EXPECT_RE <regex>   # RichLog内に正規表現マッチすること
#   DUMP_LOG
#   DUMP_BOARD
#   DUMP_SFEN           # 内部的に CMD sfen を行う
#
# コメント:
#   # で始まる行


def parse_lines(text: str) -> Iterable[Tuple[str, str]]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, *rest = line.split(" ", 1)
        arg = rest[0] if rest else ""
        yield head.upper(), arg


# -----------------------------
# RichLog extraction (robust)
# -----------------------------
def _to_plain(x: Any) -> str:
    plain = getattr(x, "plain", None)
    if plain is not None:
        return str(plain)
    try:
        return str(x)
    except Exception:
        return ""


def extract_richlog_text(logw) -> str:
    """
    RichLogの中身を、可能な限り確実にテキスト化する。
    環境によって export_text が無い/空になる場合があるので複数経路で吸う。
    """
    # 1) export_text()
    try:
        txt = logw.export_text()
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass

    # 2) lines / _lines
    for attr in ("lines", "_lines"):
        try:
            xs = getattr(logw, attr, None)
            if xs:
                out = [_to_plain(x) for x in xs]
                txt2 = "\n".join([s for s in out if s is not None])
                if txt2.strip():
                    return txt2
        except Exception:
            pass

    # 3) renderable / _content（環境差）
    for attr in ("renderable", "_content"):
        try:
            x = getattr(logw, attr, None)
            if x:
                s = _to_plain(x)
                if s.strip():
                    return s
        except Exception:
            pass

    # repr はノイズになるので空を返す
    return ""


# -----------------------------
# JSONL
# -----------------------------
def jsonl_write(fp, obj: dict) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def normalize_scenario_path(p: Path) -> Path:
    """
    引数がディレクトリなら tests/smoke_scenario.txt を優先して探す。
    """
    if p.is_dir():
        cand = p / "smoke_scenario.txt"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"scenario dir given but smoke_scenario.txt not found: {p}")
    return p


def board_plain(app) -> str:
    board = app.query_one("#board")
    r = board.render()
    return getattr(r, "plain", None) or str(r)


# -----------------------------
# Runner
# -----------------------------
@dataclass
class RunnerConfig:
    scenario_path: Path
    out_path: Path
    jsonl_path: Optional[Path]
    stop_on_fail: bool = True


async def run_scenario(app: ShogiTui, steps: list[Tuple[str, str]], cfg: RunnerConfig) -> None:
    logs: list[str] = []

    jsonl_fp = cfg.jsonl_path.open("w", encoding="utf-8") if cfg.jsonl_path else None

    def log(msg: str) -> None:
        logs.append(msg)

    def jlog(obj: dict) -> None:
        if jsonl_fp:
            jsonl_write(jsonl_fp, obj)

    async with app.run_test() as pilot:
        jlog({"ts": now_iso(), "type": "start", "scenario": str(cfg.scenario_path)})
        log("[SMOKE] start")

        for i, (op, arg) in enumerate(steps, start=1):
            log(f"[STEP {i:03d}] {op} {arg}".rstrip())
            jlog({"ts": now_iso(), "type": "step", "i": i, "op": op, "arg": arg})

            try:
                if op == "WAIT":
                    ms = int(arg)
                    await pilot.pause(ms / 1000.0)

                elif op == "KEY":
                    await pilot.press(arg)
                    await pilot.pause(0)

                elif op == "CMD":
                    inp = pilot.app.query_one("#cmd")
                    inp.value = arg
                    await pilot.press("enter")
                    await pilot.pause(0)

                elif op == "EXPECT":
                    logw = pilot.app.query_one("#log")
                    txt = extract_richlog_text(logw)
                    if arg not in txt:
                        raise AssertionError(
                            f"EXPECT failed: '{arg}' not found in log\n---LOG---\n{txt}"
                        )

                elif op == "EXPECT_RE":
                    logw = pilot.app.query_one("#log")
                    txt = extract_richlog_text(logw)
                    if not re.search(arg, txt, flags=re.MULTILINE):
                        raise AssertionError(
                            f"EXPECT_RE failed: /{arg}/ not matched\n---LOG---\n{txt}"
                        )

                elif op == "DUMP_LOG":
                    logw = pilot.app.query_one("#log")
                    txt = extract_richlog_text(logw)
                    log("[LOG]\n" + txt)
                    jlog({"ts": now_iso(), "type": "dump_log", "i": i, "log": txt})

                elif op == "DUMP_BOARD":
                    plain = board_plain(pilot.app)
                    log("[BOARD]\n" + plain)
                    jlog({"ts": now_iso(), "type": "dump_board", "i": i, "board": plain})

                elif op == "DUMP_SFEN":
                    inp = pilot.app.query_one("#cmd")
                    inp.value = "sfen"
                    await pilot.press("enter")
                    await pilot.pause(0)
                    # sfen自体はアプリがログに出す想定なので、必要なら DUMP_LOG で回収

                else:
                    raise ValueError(f"unknown op: {op}")

                jlog({"ts": now_iso(), "type": "ok_step", "i": i})

            except Exception as e:
                # FAIL を記録
                msg = str(e)
                log(f"[FAIL] step {i}: {op} {arg} -> {type(e).__name__}: {msg}")
                jlog(
                    {
                        "ts": now_iso(),
                        "type": "fail",
                        "i": i,
                        "op": op,
                        "arg": arg,
                        "exc": type(e).__name__,
                        "msg": msg,
                    }
                )

                # 追加デバッグ：盤面
                try:
                    plain = board_plain(pilot.app)
                    log("[BOARD@FAIL]\n" + plain)
                    jlog({"ts": now_iso(), "type": "dump_board", "i": i, "board": plain, "at": "fail"})
                except Exception:
                    pass

                # 追加デバッグ：ログ
                try:
                    logw = pilot.app.query_one("#log")
                    txt = extract_richlog_text(logw)
                    log("[LOG@FAIL]\n" + txt)
                    jlog({"ts": now_iso(), "type": "dump_log", "i": i, "log": txt, "at": "fail"})
                except Exception:
                    pass

                if cfg.stop_on_fail:
                    break

        # 最終ログ
        try:
            logw = pilot.app.query_one("#log")
            txt = extract_richlog_text(logw)
            log("[LOG@END]\n" + txt)
            jlog({"ts": now_iso(), "type": "end_log", "log": txt})
        except Exception:
            pass

        jlog({"ts": now_iso(), "type": "end"})

    if jsonl_fp:
        jsonl_fp.close()

    cfg.out_path.write_text("\n".join(logs), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=str, help="scenario file path, or a directory containing smoke_scenario.txt")
    ap.add_argument("--out", type=str, default="smoke.log", help="human-readable log output")
    ap.add_argument("--jsonl", type=str, default="smoke.jsonl", help="machine-readable JSONL output (empty to disable)")
    ap.add_argument("--no-stop", action="store_true", help="do not stop on first failure (best-effort)")
    args = ap.parse_args()

    scenario_path = normalize_scenario_path(Path(args.scenario))
    out_path = Path(args.out)

    jsonl_path: Optional[Path]
    if args.jsonl is None or str(args.jsonl).strip() == "":
        jsonl_path = None
    else:
        jsonl_path = Path(args.jsonl)

    steps = list(parse_lines(scenario_path.read_text(encoding="utf-8")))
    cfg = RunnerConfig(
        scenario_path=scenario_path,
        out_path=out_path,
        jsonl_path=jsonl_path,
        stop_on_fail=(not args.no_stop),
    )

    app = ShogiTui()
    asyncio.run(run_scenario(app, steps, cfg))
    print(f"Wrote: {out_path}")
    if jsonl_path:
        print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()
