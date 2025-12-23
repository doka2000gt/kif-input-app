from __future__ import annotations
from typing import Dict, List, Optional
import pathlib
import time
import re   # 変化ブロックのprefix作成で re.match を使ってるので必要

from paths import _ensure_output_dir
from solver_core import solve_mate_tree, prune_tree_to_max_leaves, enumerate_solution_paths
from kif_format import generate_kif_single_line
from kif_parser import parse_kif_startpos_piyo, parse_kif_moves_and_variations
from sfen import compute_gote_remaining, snapshot_to_sfen   # ←もし既に別ファイルならそれを。solver内ならそこから。
from models import SolveLimits


def _basename_no_ext(p: str) -> str:
    return pathlib.Path(p).stem

def batch_process_kif(path: str, default_ply: Optional[int] = None,
                      limits: Optional[SolveLimits] = None,
                      check_only: bool = True) -> List[str]:
    """
    Process one kif:
    - If it contains '変化：' blocks: split each branch (main + each variation) into separate files.
      (We do not attempt to re-solve; we just split text-based branches by reusing our own writer.)
    - If it has no variations: try to parse startpos, then solve for mates within ply (from file or default_ply),
      and write each mate line as separate KIF.
      If no mate found within ply: write the original file as OUTPUT/<base>.kif.
    Returns written file paths.
    """
    outdir = _ensure_output_dir()
    raw = pathlib.Path(path).read_bytes()
    try:
        txt = raw.decode("cp932", errors="replace")
    except Exception:
        txt = raw.decode("utf-8", errors="replace")

    main_lines, variations, ply_guess = parse_kif_moves_and_variations(txt)
    base = _basename_no_ext(pathlib.Path(path).name)

    written: List[str] = []

    # Case A: has variations -> split (best-effort; use original text blocks)
    if variations:
        # Write main as _001, then each variation as _002... by reusing original text but stripping other variations
        # This preserves maximum ANKIF compatibility because it's basically the same format.
        def _write_text(outname: str, body_lines: List[str]) -> str:
            # keep header up to moves header line, then body_lines, then end
            lines = txt.splitlines()
            # keep everything until moves header (inclusive)
            keep = []
            for ln in lines:
                keep.append(ln)
                if ln.strip().startswith("手数----指手"):
                    break
            out_txt = "\n".join(keep + body_lines) + "\n"
            p = outdir / outname
            p.write_bytes(out_txt.encode("cp932", errors="replace"))
            return str(p)

        written.append(_write_text(f"{base}.kif", main_lines))
        for j, (sp, mvlines) in enumerate(variations, start=2):
            # prepend main up to sp-1 from main_lines (text-based; assume numbering matches)
            prefix = []
            for ln in main_lines:
                m = re.match(r"\s*(\d+)\s", ln)
                if m and int(m.group(1)) >= sp:
                    break
                prefix.append(ln)
            combined = prefix + mvlines
            written.append(_write_text(f"{base}_{j:03d}.kif", combined))
        return written

    # Case B: no variations -> solve
    try:
        board_map, hands_b, stm = parse_kif_startpos_piyo(txt)
    except Exception as e:
        # cannot parse -> just copy original
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: 開始局面の解析に失敗（{e}）。元KIFをコピーしました: {dst}")
        return [str(dst)]

    # Determine ply
    ply = default_ply if default_ply is not None else (ply_guess if ply_guess else 9)
    if ply % 2 == 0:
        ply += 1  # tsume typically odd for attacker to finish

    try:
        import shogi as pyshogi
    except Exception:
        # no python-shogi -> just copy
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: python-shogi がないため解探索はスキップ。元KIFをコピー: {dst}")
        return [str(dst)]

    gote_auto = compute_gote_remaining(board_map, hands_b)
    sfen = snapshot_to_sfen(board_map, hands_b, stm, gote_auto)
    board0 = pyshogi.Board(sfen)

    stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}
    tree = solve_mate_tree(board0, ply, attacker_turn=pyshogi.BLACK, check_only=check_only, stats=stats, limits=limits)
    elapsed = time.perf_counter() - float(stats.get("start", 0.0))
    if tree is None or not tree.children:
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: {ply}手以内の詰み筋が見つからず（nodes={stats['nodes']} {elapsed:.3f}s）。元KIFをコピー: {dst}")
        return [str(dst)]

    # produce all solution move-paths from tree
    tree = prune_tree_to_max_leaves(tree, limits.max_solutions if limits else 300)
    paths = enumerate_solution_paths(tree)
    if not paths:
        dst = outdir / f"{base}.kif"
        dst.write_bytes(raw)
        print(f"[batch] {path}: 解答筋が空でした（nodes={stats['nodes']} {elapsed:.3f}s）。元KIFをコピー: {dst}")
        return [str(dst)]

    # Write: first solution -> base.kif, others -> base_002.kif ...
    written: List[str] = []
    seen_kif: Dict[str,str] = {}
    p0 = generate_kif_single_line(board0, hands_b, gote_auto, "", "", paths[0], f"{base}.kif", seen_kif)
    if p0:
        written.append(str(p0))
    for j, mvlist in enumerate(paths[1:], start=2):
        fname = f"{base}_{j:03d}.kif"
        pw = generate_kif_single_line(board0, hands_b, gote_auto, "", "", mvlist, fname, seen_kif)
        if pw:
            written.append(str(pw))

    cutoff_note = "（制限で打ち切り・部分解の可能性）" if stats.get("cutoff") else ""
    print(f"[batch] {path}: 解答筋={len(paths)} {cutoff_note} / 探索={elapsed:.3f}s nodes={stats['nodes']} -> OUTPUT")
    return written

def batch_process_path(path: str, default_ply: Optional[int] = None,
                       limits: Optional[SolveLimits] = None,
                       check_only: bool = True) -> List[str]:
    """
    Process a folder or one file.
      batch <dir> [ply]
      batch <file.kif> [ply]
    """
    p = pathlib.Path(path)
    written_all: List[str] = []
    if p.is_dir():
        for kif in sorted(p.glob("*.kif")):
            written_all.extend(batch_process_kif(str(kif), default_ply=default_ply, limits=limits, check_only=check_only))
    else:
        written_all.extend(batch_process_kif(str(p), default_ply=default_ply, limits=limits, check_only=check_only))
    return written_all
