# tests/test_batch_solve_outputs.py
import importlib.util
from pathlib import Path
import shutil

def import_solver_from_tmp(tmp_path: Path):
    """
    テスト用に solver と paths を tmp にコピーして import する。
    __file__ が tmp を指すので、OUTPUT/INPUT も tmp 配下になり安全。
    """
    import sys
    import shutil
    import importlib.util

    root = Path(__file__).resolve().parents[1]  # project root

    # 1) solver と 依存モジュールを tmp にコピー
    #    （tmp 上で import されるため、依存はすべてここに列挙する）
    src_solver    = root / "kif_tsume_cui_solver.py"
    src_paths     = root / "paths.py"
    src_constants = root / "constants.py"
    src_models    = root / "models.py"
    src_helpers = root / "helpers.py"
    src_kif_format = root / "kif_format.py"
    src_position = root / "position.py"
    src_manual_kif = root / "manual_kif.py"

    assert src_solver.exists(),    f"not found: {src_solver}"
    assert src_paths.exists(),     f"not found: {src_paths}"
    assert src_constants.exists(), f"not found: {src_constants}"
    assert src_models.exists(),    f"not found: {src_models}"
    assert src_helpers.exists(), f"not found: {src_helpers}"
    assert src_kif_format.exists(), f"not found: {src_kif_format}"
    assert src_position.exists(), f"not found: {src_position}"
    assert src_manual_kif.exists(), f"not found: {src_manual_kif}"

    dst_solver    = tmp_path / "kif_tsume_cui_solver.py"
    dst_paths     = tmp_path / "paths.py"
    dst_constants = tmp_path / "constants.py"
    dst_models    = tmp_path / "models.py"
    dst_helpers = tmp_path / "helpers.py"
    dst_kif_format = tmp_path / "kif_format.py"
    dst_position = tmp_path / "position.py"
    dst_manual_kif = tmp_path / "manual_kif.py"

    shutil.copy2(src_solver,    dst_solver)
    shutil.copy2(src_paths,     dst_paths)
    shutil.copy2(src_constants, dst_constants)
    shutil.copy2(src_models,    dst_models)
    shutil.copy2(src_helpers, dst_helpers)
    shutil.copy2(src_kif_format, dst_kif_format)
    shutil.copy2(src_position, dst_position)
    shutil.copy2(src_manual_kif, dst_manual_kif)

    # 2) tmp を import 検索パス先頭に入れる（paths を解決するため）
    sys.path.insert(0, str(tmp_path))

    # 3) solver を import
    module_name = "solver_under_test"
    spec = importlib.util.spec_from_file_location(module_name, dst_solver)
    assert spec and spec.loader

    mod = importlib.util.module_from_spec(spec)

    # ★重要：dataclass の型注釈文字列解決のため先に登録
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)
    return mod



def list_output_files(mod, stem: str):
    outdir = mod._ensure_output_dir()
    return sorted(outdir.glob(f"{stem}*.kif"))

def test_batch_process_kif_generates_branches(tmp_path):
    mod = import_solver_from_tmp(tmp_path)

    # python-shogi が無い環境ではスキップ
    if not getattr(mod, "HAS_PYSHOGI", False):
        import pytest
        pytest.skip("python-shogi not installed; skip")

    # tests/data/demo.kif を INPUT にコピーして使う（ゴールデンマスター方式）
    root = Path(__file__).resolve().parents[1]
    src_demo = root / "tests" / "data" / "demo.kif"
    assert src_demo.exists(), f"not found: {src_demo}"

    inp = mod._ensure_input_dir()
    demo_path = inp / "demo.kif"
    demo_path.write_bytes(src_demo.read_bytes())

    # 3手（あなたの実績に合わせる）
    limits = mod.SolveLimits(max_nodes=50000, max_time_sec=5.0, max_solutions=300)
    written = mod.batch_process_kif(str(demo_path), default_ply=3, limits=limits, check_only=True)

    assert len(written) == 6, f"expected 6 outputs, got {len(written)}: {written}"

    outdir = mod._ensure_output_dir()
    outs = sorted(outdir.glob("demo*.kif"))
    names = [p.name for p in outs]

    assert "demo.kif" in names
    assert "demo_002.kif" in names
    assert "demo_003.kif" in names
    assert "demo_004.kif" in names
    assert "demo_005.kif" in names
    assert "demo_006.kif" in names

def test_format_total_time_basic(tmp_path):
    mod = import_solver_from_tmp(tmp_path)
    assert mod.format_total_time(0) == "00:00:00"
    assert mod.format_total_time(9) == "00:00:09"
    assert mod.format_total_time(75) == "00:01:15"
    assert mod.format_total_time(3671) == "01:01:11"
