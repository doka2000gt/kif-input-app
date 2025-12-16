diff --git a/paths.py b/paths.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/paths.py
@@ -0,0 +1,62 @@
+#!/usr/bin/env python3
+# -*- coding: utf-8 -*-
+"""
+paths.py
+
+入出力フォルダ（INPUT/OUTPUT）とパス解決の共通化。
+モジュール化の最初の一歩として切り出す。
+"""
+
+from __future__ import annotations
+
+from pathlib import Path
+
+
+def _base_dir() -> Path:
+    # このモジュール（paths.py）のあるディレクトリを基準にする
+    return Path(__file__).resolve().parent
+
+
+def _output_dir() -> Path:
+    return _base_dir() / "OUTPUT"
+
+
+def _ensure_output_dir() -> Path:
+    out = _output_dir()
+    out.mkdir(parents=True, exist_ok=True)
+    return out
+
+
+def _input_dir() -> Path:
+    return _base_dir() / "INPUT"
+
+
+def _ensure_input_dir() -> Path:
+    inp = _input_dir()
+    inp.mkdir(parents=True, exist_ok=True)
+    return inp
+
+
+# 起動時に標準フォルダを作成（従来挙動を維持）
+INPUT_DIR = _ensure_input_dir()
+OUTPUT_DIR = _ensure_output_dir()
+
+
+def _resolve_kif_path(name: str) -> Path:
+    """相対パス（ディレクトリ無し）の場合は OUTPUT/ に保存する。"""
+    p = Path(name)
+    if p.is_absolute() or p.parent != Path("."):
+        return p
+    return _ensure_output_dir() / p.name
+
+
+def _resolve_existing_kif_path(name: str) -> Path:
+    """読み込み用。相対パスで見つからなければ OUTPUT/ も探す。"""
+    p = Path(name)
+    if p.exists():
+        return p
+    if not p.is_absolute() and p.parent == Path("."):
+        q = _output_dir() / p.name
+        if q.exists():
+            return q
+    return p
diff --git a/kif_tsume_cui_solver.py b/kif_tsume_cui_solver.py
index 2222222..3333333 100644
--- a/kif_tsume_cui_solver.py
+++ b/kif_tsume_cui_solver.py
@@ -29,46 +29,17 @@ import datetime
 import time
 import hashlib
 
 from pathlib import Path
-
-def _output_dir() -> Path:
-    # OUTPUT directory next to this script
-    return Path(__file__).resolve().parent / "OUTPUT"
-
-def _ensure_output_dir() -> Path:
-    out = _output_dir()
-    out.mkdir(parents=True, exist_ok=True)
-    return out
-
-def _input_dir() -> Path:
-    # INPUT directory next to this script
-    return Path(__file__).resolve().parent / "INPUT"
-
-def _ensure_input_dir() -> Path:
-    inp = _input_dir()
-    inp.mkdir(parents=True, exist_ok=True)
-    return inp
-
-
-# Create standard folders on startup
-INPUT_DIR = _ensure_input_dir()
-OUTPUT_DIR = _ensure_output_dir()
-
-def _resolve_kif_path(name: str) -> Path:
-    """If name is relative (no directory), store into OUTPUT/."""
-    p = Path(name)
-    if p.is_absolute() or p.parent != Path('.'):
-        return p
-    return _ensure_output_dir() / p.name
-
-def _resolve_existing_kif_path(name: str) -> Path:
-    """Resolve a KIF path for reading. If relative and not found, try OUTPUT/."""
-    p = Path(name)
-    if p.exists():
-        return p
-    if not p.is_absolute() and p.parent == Path('.'):
-        q = _output_dir() / p.name
-        if q.exists():
-            return q
-    return p
-
-import re
+
+from paths import (
+    _input_dir,
+    _ensure_input_dir,
+    _output_dir,
+    _ensure_output_dir,
+    INPUT_DIR,
+    OUTPUT_DIR,
+    _resolve_kif_path,
+    _resolve_existing_kif_path,
+)
+
+import re
 
 # -------- Optional library: python-shogi --------
 HAS_PYSHOGI = False
@@ -813,12 +784,7 @@ def parse_kif_moves_and_variations(text: str):
                 ply = max(ply, int(m.group(1)))
     return main, vars_, ply
 
-def _ensure_output_dir() -> Path:
-    outdir = _output_dir()
-    outdir.mkdir(parents=True, exist_ok=True)
-    return outdir
-
 def _basename_no_ext(p: str) -> str:
     return pathlib.Path(p).stem
