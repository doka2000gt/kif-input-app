#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Failure:
    step: int
    op: str
    arg: str
    exc: str
    msg: str
    board: Optional[str] = None
    log: Optional[str] = None


def load_failure_from_jsonl(path: Path) -> Optional[Failure]:
    board_at_fail = None
    log_at_fail = None
    last_board = None
    last_log = None

    fail = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        t = obj.get("type")

        if t == "dump_board":
            last_board = obj.get("board")
        elif t == "dump_log":
            last_log = obj.get("log")
        elif t == "fail":
            fail = Failure(
                step=int(obj.get("i", -1)),
                op=str(obj.get("op")),
                arg=str(obj.get("arg")),
                exc=str(obj.get("exc")),
                msg=str(obj.get("msg")),
                board=last_board,
                log=last_log,
            )
            break

    return fail


def classify(f: Failure) -> tuple[str, list[str]]:
    """症状カテゴリと提案の叩き台（ルールベース）"""
    txt = (f.msg or "") + "\n" + (f.log or "")

    # 1) EXPECT失敗
    if "EXPECT failed" in txt:
        return (
            "期待ログ不一致（ログ採取/文言変更/待機不足）",
            [
                "runnerのログ採取が空になっていないか（export_text/lines）を確認",
                "CMD直後に WAIT を増やす（50ms→200msなど）",
                "EXPECT文字列を固定語（例: '手番=' 'SFEN' '[OK]'）に寄せる",
            ],
        )

    # 2) ピッカーが勝手に開く/Enter回り込み
    if "screen_stack" in txt or "Modal" in txt or "picker" in txt.lower():
        return (
            "画面遷移/Enter回り込み（フォーカス競合）",
            [
                "Enterでピッカーを開く処理を call_later にし、直前Enterの余波を遮断",
                "INPUT欄の on_input_submitted が走る条件を mode==INPUT に厳格化",
                "BINDINGSの衝突（h等）を解消し、NORMALとINPUTで処理を分離",
            ],
        )

    # 3) import/パス
    if "ModuleNotFoundError" in txt or "ImportError" in txt:
        return (
            "import/パス問題",
            [
                "make smoke 内で PYTHONPATH=. を設定しているか確認",
                "tests/__init__.py の有無と -m 実行の整合を確認",
            ],
        )

    # 4) 属性/型の不一致
    if "has no attribute" in txt:
        return (
            "モデル差異（属性名/型が混在）",
            [
                "盤面に入れる Piece を models.Piece に統一（自前Piece混在を禁止）",
                "表示/変換は getattr で吸収しつつ、最終的にはデータ構造を一本化",
            ],
        )

    # 5) unpack 要素数
    if "not enough values to unpack" in txt:
        return (
            "snapshotの戻り値仕様差異（3要素/4要素）",
            [
                "sfen_to_snapshotの戻り値 len==3/4 を吸収する分岐を追加",
                "start/load で moves を確実に初期化する",
            ],
        )

    return (
        "未分類（ログ断片から手動解析）",
        [
            "FAIL step と直前の [LOG]/[BOARD] を確認し、再現条件を特定",
            "分類ルールを追加して analyzer を育てる（今回のケースをパターン登録）",
        ],
    )


def render_report(f: Failure) -> str:
    cat, advice = classify(f)
    out = []
    out.append("=== Smoke Analyze Report ===")
    out.append(f"FAIL step: {f.step}")
    out.append(f"op: {f.op} {f.arg}".rstrip())
    out.append(f"exception: {f.exc}")
    out.append(f"message: {f.msg}")
    out.append("")
    out.append(f"[category] {cat}")
    out.append("")
    out.append("[suggestions]")
    for i, a in enumerate(advice, 1):
        out.append(f"  {i}. {a}")

    if f.board:
        out.append("")
        out.append("[board@fail]")
        out.append(f.board)

    if f.log:
        out.append("")
        out.append("[log@fail]")
        out.append(f.log)

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="smoke.jsonl")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"jsonl not found: {path}")

    fail = load_failure_from_jsonl(path)
    if not fail:
        print("=== Smoke Analyze Report ===")
        print("No failure found (PASS or jsonl missing fail entry).")
        return

    print(render_report(fail))


if __name__ == "__main__":
    main()
