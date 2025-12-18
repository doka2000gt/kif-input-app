#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
"""
kif_tsume_cui_ankif_v4.py

Tsume-shogi focused CUI tool:
- Create start position in CUI (board + hands + side-to-move)
- Fast numeric input for moves (move / promote / drop-pick)
- KIF output tuned to "Kifu for Mac V0.53" style (ANKIF-friendly)
- Optional python-shogi integration (recommended):
    pip install python-shogi
  Enables:
    * strict legality / checkmate detection
    * solve command: enumerate forced mate lines up to N plies (odd number, e.g., 1/3/5/7/9)

Notes:
- We intentionally focus on tsume usage. "Hirate" etc. can be added later as separate modes.
"""

from typing import Dict, List, Optional, Tuple
from models import Piece, Move, SolveNode, SolveLimits
import copy
import json
import pathlib
import datetime
import time
import hashlib
import os

from constants import (
    FW_DIGITS,
    RANK_KANJI,
    PIECE_JP,
    PROMOTED_JP,
    PROMOTABLE,
    TOTAL_COUNTS,
    KIND_TO_PYO,
    PIECE_LEGEND,
)

from paths import (
    _ensure_input_dir,
    _ensure_output_dir,
    _resolve_kif_path,
    _resolve_existing_kif_path,
)

from helpers import (
    format_total_time,
    now_yyyy_mm_dd_hhmmss,
    sq_to_kif,
    sq_to_paren,
    inv_count_kanji,
    parse_piece_token,
    build_piece_menu,
    _dedup_key_from_kif_text,
    _write_kif_unique,
)

from kif_format import (
    kif_line_for_shogi_move,
    board0_to_piyo,
    _hands_to_line,
    generate_kif_single_line,
    generate_kif_with_variations,
    build_mainline_and_variations,
    emit_lines_for_branch,
)

from manual_kif import (
    board_map_to_piyo,
    kif_line_for_minimal_move,
    apply_minimal_to_tmp,
)

from position import ShogiPosition

from manual_kif import (
    board_map_to_piyo,
    kif_line_for_minimal_move,
    apply_minimal_to_tmp,
)

from help_texts import HELP_MAIN, HELP_MAP

from solver_core import (
    solve_mate_tree,
    count_solutions,
    prune_tree_to_max_leaves,
    enumerate_solution_paths,
)

from batch_runner import (
    batch_process_kif,
    batch_process_path,
)

from sfen import (
    compute_gote_remaining,
    snapshot_to_sfen,
    sfen_to_snapshot,   # 使っているなら
)


import re

# -------- Optional library: python-shogi --------
try:
    import shogi  # pip install python-shogi  (import name is "shogi")
    HAS_PYSHOGI = True
except Exception:
    shogi = None
    HAS_PYSHOGI = False

def save_startpos_file(filename: str, board_map, hands_b, side_to_move: str, sente: str, gote: str):
    gote_auto = compute_gote_remaining(board_map, hands_b)
    sfen = snapshot_to_sfen(board_map, hands_b, side_to_move, gote_auto)
    data = {
        "format": "kif_tsume_cui_startpos_v1",
        "sfen": sfen,
        "sente": sente,
        "gote": gote,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_startpos_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    sfen = data.get("sfen")
    if not sfen:
        raise ValueError("startposファイルにsfenがありません")
    board_map, hands_b, side_to_move = sfen_to_snapshot(sfen)
    sente = data.get("sente")
    gote = data.get("gote")
    return board_map, hands_b, side_to_move, sente, gote

def preview_kif_file(filename: str, max_lines: int = 120):
    p = _resolve_existing_kif_path(filename)
    try:
        raw = pathlib.Path(p).read_bytes()
        txt = raw.decode("cp932", errors="replace")
    except Exception:
        txt = pathlib.Path(filename).read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    n = min(len(lines), max_lines)
    print(f"[preview] {p}（全{len(lines)}行） 先頭{n}行を表示します")
    for i in range(n):
        print(lines[i])
    if len(lines) > n:
        print(f"...（残り {len(lines)-n} 行）")

def clear_screen() -> None:
    # Windows: cls / macOS+Linux: clear
    os.system("cls" if os.name == "nt" else "clear")

# ----------------- Main -----------------
def main():
    clear_screen()
    pos = ShogiPosition()
    start_snapshot = None  # (board, hands_b, side_to_move)
    last_piece_token: Optional[str] = None
    end_result: Optional[str] = None  # "詰み" or "投了" (manual override)

    sente = input("先手名（Enterで先手）: ").strip() or ""
    gote  = input("後手名（Enterで後手）: ").strip() or ""
    print("\n" + HELP_MAIN)

    def show():
        print(f"\n手番: {'先手(▲)' if pos.side_to_move=='B' else '後手(△)'}")
        # show gote remaining preview (based on current board & sente hand)
        gote_auto = compute_gote_remaining(pos.board, pos.hands["B"])
        print(f"後手の持駒：{_hands_to_line(gote_auto)}")
        print(pos.board_to_piyo())
        print(f"先手の持駒：{pos.hands_to_piyo('B')}")
        print("ガイド: help solve / p 55 / h b P 2 / turn b / start / 7776 / 076 / s out.kif / solve 9 out.kif\n")
        print(PIECE_LEGEND)

    while True:
        prompt = "▲ " if pos.side_to_move == "B" else "△ "
        s = input(prompt).strip()
        if not s:
            continue

        if s == "q":
            break

        if s in ("new", "reset"):
            pos.clear_all()
            start_snapshot = None
            last_piece_token = None
            end_result = None

            # 名前もリセットしたいなら（任意）
            ans = input("先手/後手名もリセットしますか？ (y/N): ").strip().lower()
            if ans == "y":
                sente = input("先手名（Enterで先手）: ").strip() or ""
                gote  = input("後手名（Enterで後手）: ").strip() or ""

            print("OK: 新しい局面入力を開始します（p/h/turn → start）")
            continue

        if s.startswith("help"):
            t = s.split()
            if len(t) == 1:
                print(HELP_MAIN)
            else:
                topic = t[1].lower()
                print(HELP_MAP.get(topic, f"[help] topic '{topic}' は未対応です。使えるtopic: {', '.join(HELP_MAP.keys())}"))
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

        # end command (manual)
        if s.startswith("end"):
            t = s.split()
            if len(t) == 1:
                print("終局を選んでください: 1)詰み  2)投了")
                sel = input("選択: ").strip()
                if sel == "1":
                    end_result = "詰み"
                elif sel == "2":
                    end_result = "投了"
                else:
                    print("キャンセル")
                    continue
                print(f"OK: 終局={end_result}")
                continue
            if len(t) == 2:
                if t[1] in ("mate", "tsumi", "詰み"):
                    end_result = "詰み"
                    print("OK: 終局=詰み")
                elif t[1] in ("resign", "toryo", "投了"):
                    end_result = "投了"
                    print("OK: 終局=投了")
                else:
                    print("end mate / end resign")
                continue
            print("end / end mate / end resign")
            continue

        # p command
        if s.startswith("p "):
            t = s.split()
            if len(t) not in (2,3):
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

            # menu
            cur = pos.board[(f,r)]
            cur_label = "・" if cur is None else (("△" if cur.color=="W" else "▲") + KIND_TO_PYO.get((cur.kind, cur.prom), PIECE_JP[cur.kind]))
            print(f"[{f}{r}] 現在: {cur_label}")

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

        # start snapshot
        if s == "start":
            start_snapshot = (copy.deepcopy(pos.board), copy.deepcopy(pos.hands["B"]), pos.side_to_move)
            pos.clear_moves()
            end_result = None
            print("OK: この局面を開始局面として確定しました（ここから手順入力）")
            continue

        # undo
        if s == "u":
            if pos.undo():
                print("OK: 1手戻しました")
            else:
                print("戻せる手がありません")
            continue

        # solve command

        # start-position quick save/load (ready-to-start)
        if s.startswith("possave "):
            fn = s[len("possave "):].strip()
            if not fn.lower().endswith(".pos"):
                fn += ".pos"
            try:
                save_startpos_file(fn, pos.board, pos.hands["B"], pos.side_to_move, sente, gote)
                print(f"保存しました（開始局面）: {fn}")
            except Exception as e:
                print(f"保存エラー: {e}")
            continue

        if s.startswith("posload "):
            fn = s[len("posload "):].strip()
            try:
                board_map, hands_b, stm, se2, go2 = load_startpos_file(fn)
                pos.clear_all()
                pos.board = board_map
                pos.hands["B"] = hands_b
                pos.hands["W"] = {}  # not used (auto)
                pos.side_to_move = stm
                # set start snapshot immediately (so user can solve right away)
                start_snapshot = (copy.deepcopy(pos.board), copy.deepcopy(pos.hands["B"]), pos.side_to_move)
                end_result = None
                if se2: sente = se2
                if go2: gote = go2
                print(f"読み込みました（開始局面・start済み）: {fn}")
            except Exception as e:
                print(f"読み込みエラー: {e}")
            continue

        # KIF preview
        if s.startswith("preview "):
            t = s.split()
            if len(t) not in (2,3):
                print("形式: preview file.kif [lines]")
                continue
            fn = t[1]
            n = 120
            if len(t) == 3:
                try: n = int(t[2])
                except: pass
            try:
                preview_kif_file(fn, max_lines=n)
            except Exception as e:
                print(f"previewエラー: {e}")
            continue


        # Batch process existing KIFs: split variations or solve-and-split
        # Usage:
        #   batch INPUT_DIR [ply]
        #   batch somefile.kif [ply]
        # Notes:
        #   - if the input KIF contains '変化：' blocks, we split them into separate files.
        #   - if not, we solve for multiple mate lines within ply (default: infer from file, else 9).
        if s.startswith("batch"):
            t = s.split()
            # デフォルトは INPUT フォルダ一括
            target = None
            ply = None
            if len(t) == 1:
                target = str(_ensure_input_dir())
            elif len(t) == 2:
                # batch 9 のように数字だけ来たら ply とみなす
                try:
                    ply = int(t[1])
                    target = str(_ensure_input_dir())
                except:
                    target = t[1]
            else:
                # batch <dir|file.kif> [ply]
                target = t[1]
                try:
                    ply = int(t[2])
                except:
                    ply = None

            # limits use the same defaults as solve (安全弁)
            limits = SolveLimits(max_nodes=50000, max_time_sec=5.0, max_solutions=300)
            try:
                written = batch_process_path(target, default_ply=ply, limits=limits, check_only=True)
            except Exception as e:
                print(f"[batch] エラー: {e}")
                print("形式: batch [dir|file.kif|ply] [ply]")
                print("  例: batch            # INPUT 内を一括処理")
                print("      batch 9          # INPUT 内を 9 手で一括処理")
                print("      batch INPUT 9    # INPUT フォルダを 9 手で一括処理")
                print("      batch some.kif 3 # 単体KIFを 3 手で探索/分割")
                continue

            if not written:
                print("[batch] 対象が見つかりません（.kif が無いか、フォルダ/ファイルが存在しません）")
                continue
            print("[batch] 出力しました:")
            for w in written:
                print("  " + w)
            continue

        if s.startswith("solve"):
            if not HAS_PYSHOGI:
                print("python-shogi が必要です: pip install python-shogi")
                continue
            if start_snapshot is None:
                print("先に start で開始局面を確定してください（局面作成→start→solve）")
                continue

            t = s.split()
            if len(t) < 2:
                print("形式: solve 9 [out.kif] [--maxnodes N] [--maxtime SEC] [--maxsol N]")
                continue

            # parse args
            try:
                ply = int(t[1])
            except:
                print("形式: solve 9 [out.kif] ... （手数は整数）")
                continue

            out = None
            i = 2
            if i < len(t) and not t[i].startswith("--"):
                out = t[i]
                i += 1
                if out and not out.lower().endswith(".kif"):
                    out += ".kif"

            # defaults
            maxnodes = 2000000
            maxtime = 120
            maxsol = 20000

            # flags
            while i < len(t):
                if t[i] == "--maxnodes" and i + 1 < len(t):
                    maxnodes = int(t[i+1]); i += 2; continue
                if t[i] == "--maxtime" and i + 1 < len(t):
                    maxtime = float(t[i+1]); i += 2; continue
                if t[i] == "--maxsol" and i + 1 < len(t):
                    maxsol = int(t[i+1]); i += 2; continue
                print(f"[solve] 不明なオプション: {t[i]}")
                print("形式: solve 9 [out.kif] [--maxnodes N] [--maxtime SEC] [--maxsol N]")
                break
            else:
                # ok
                pass

            if ply <= 0:
                print("手数は正の整数で指定してください（例: solve 9）")
                continue

            board_map, hands_b, stm = start_snapshot
            gote_auto = compute_gote_remaining(board_map, hands_b)
            sfen = snapshot_to_sfen(board_map, hands_b, stm, gote_auto)

            b = shogi.Board(sfen)
            attacker_turn = b.turn  # side to move at start

            limits = SolveLimits(max_nodes=maxnodes, max_time_sec=maxtime, max_solutions=maxsol)
            stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}

            t0 = time.perf_counter()
            tree = solve_mate_tree(
                b, ply_left=ply, attacker_turn=attacker_turn, check_only=True, memo={}, stats=stats, limits=limits
            )
            elapsed = time.perf_counter() - t0

            if tree is None or not tree.children:
                msg = f"[solve] {ply}手以内の強制詰みは見つかりませんでした"
                if stats.get("cutoff"):
                    msg += "（制限で打ち切り）"
                print(msg)
                print(f"[solve] 探索: {elapsed:.3f}s / nodes={stats.get('nodes',0)}")
                continue

            # prune for output safety
            tree_out = prune_tree_to_max_leaves(tree, maxsol)
            sol = count_solutions(tree_out)

            cutoff_note = "（制限で打ち切り・部分解の可能性）" if stats.get("cutoff") else ""
            print(f"[solve] 解答筋（葉の数）: {sol} {cutoff_note}")
            print(f"[solve] 探索: {elapsed:.3f}s / nodes={stats.get('nodes',0)}")

            if out:
                # Write each solution line to separate KIF files under OUTPUT/
                board0 = shogi.Board(sfen)
                paths = enumerate_solution_paths(tree_out)
                stem = pathlib.Path(out).stem
                written: List[str] = []
                if not paths:
                    print("[solve] 解答筋がありません")
                else:
                    # First line: stem.kif, others: stem_002.kif...
                    seen_kif: Dict[str,str] = {}
                    p0 = generate_kif_single_line(board0, hands_b, gote_auto, sente, gote, paths[0], f"{stem}.kif", seen_kif)
                    if p0:
                        written.append(str(p0))
                    for i_line, mvlist in enumerate(paths[1:], start=2):
                        pw = generate_kif_single_line(board0, hands_b, gote_auto, sente, gote, mvlist, f"{stem}_{i_line:03d}.kif", seen_kif)
                        if pw:
                            written.append(str(pw))
                    if len(written) == 1:
                        print(f"[solve] KIFを OUTPUT に保存しました: {stem}.kif")
                    else:
                        print(f"[solve] KIFを {len(written)} 本、OUTPUT に保存しました: {stem}.kif + {stem}_002.kif ...")

            else:
                print("[solve] 出力ファイル名を付けると、各解答筋を別KIFでOUTPUTへ保存します。例: solve 9 solve.kif")
            continue

        # save command (manual sequence)
        if s.startswith("s "):
            fn = s[2:].strip()
            if not fn.lower().endswith(".kif"):
                fn += ".kif"
            if start_snapshot is None:
                print("先に start で開始局面を確定してください（局面作成→start→手順入力）")
                continue

            board0_map, hands0_b, stm0 = start_snapshot
            gote_auto = compute_gote_remaining(board0_map, hands0_b)

            # Try to auto-detect mate with python-shogi if available and end_result not set
            auto_end = end_result
            if auto_end is None and HAS_PYSHOGI:
                try:
                    sfen = snapshot_to_sfen(board0_map, hands0_b, stm0, gote_auto)
                    b = shogi.Board(sfen)
                    for mv in pos.moves:
                        # Convert minimal Move -> USI move string is non-trivial without strict mapping;
                        # here we simply skip auto-check in manual save path to avoid mismatches.
                        # (Solve path is the main auto path.)
                        pass
                except:
                    pass

            # Build KIF
            out: List[str] = []
            out.append("# ---- Kifu for Mac V0.53 夢の詰将棋メーカー by CUI ----")
            out.append(f"終了日時：{now_yyyy_mm_dd_hhmmss()}")
            out.append("手合割：平手")
            out.append("後手の持駒：" + _hands_to_line(gote_auto))
            out.append(_board_map_to_piyo(board0_map))
            out.append("先手の持駒：" + _hands_to_line(hands0_b))
            out.append(f"先手：{sente}")
            out.append(f"後手：{gote}")
            out.append("手数----指手---------消費時間--")

            # We continue to output the user's entered lines using our lightweight formatter (no strict legality).
            sec_per_move = 3
            total_sec = 0
            prev_to = None
            tmp_board = copy.deepcopy(board0_map)
            tmp_hands = {"B": copy.deepcopy(hands0_b), "W": copy.deepcopy(gote_auto)}
            tmp_side = stm0

            for i, mv in enumerate(pos.moves, start=1):
                total_sec += sec_per_move
                line, prev_to = _kif_line_for_minimal_move(i, tmp_board, mv, prev_to, sec_per_move, total_sec)
                out.append(line)
                # apply minimally to keep "同" consistent
                _apply_minimal_to_tmp(tmp_board, tmp_hands, tmp_side, mv)
                tmp_side = "W" if tmp_side == "B" else "B"

            # end line
            endtxt = auto_end or end_result
            if endtxt is None:
                endtxt = "詰み"  # for tsume workflow, default to mate if user forgets (can be overridden with end resign)
            out.append(f"{len(pos.moves)+1:4d} {endtxt:<12} ( 0:{sec_per_move:02d}/00:00:{total_sec:02d})")

            text = "\n".join(out) + "\n"
            outp = _resolve_kif_path(fn)
            outp.write_bytes(text.encode("cp932", errors="replace"))
            print(f"保存しました: {outp}")
            continue

        # numeric move input
        try:
            if start_snapshot is None:
                print("先に局面を作って start してください（help solve を参照）")
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
                mv = pos.apply_move_minimal(kind, None, to, False, True)
                idx = len(pos.moves)
                print(f"{idx:4d} {sq_to_kif(*mv.to_sq)}{PIECE_JP[mv.kind]}打")
                continue

            if mode == "move":
                p = pos.board.get(frm)
                if p is None or p.color != pos.side_to_move:
                    raise ValueError("移動元に手番の駒がありません")
                mv = pos.apply_move_minimal(p.kind, frm, to, promote, False)
                idx = len(pos.moves)
                suffix = "成" if promote else ""
                print(f"{idx:4d} {sq_to_kif(*mv.to_sq)}{PIECE_JP[mv.kind]}{suffix}{sq_to_paren(*mv.from_sq)}")
                continue

        except Exception as e:
            print(f"入力エラー: {e}")


if __name__ == "__main__":
    main()
