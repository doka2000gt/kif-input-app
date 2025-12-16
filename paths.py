#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paths.py

入出力フォルダ（INPUT/OUTPUT）とパス解決の共通化。
モジュール化の最初の一歩として切り出す。
"""

from __future__ import annotations

from pathlib import Path


def _base_dir() -> Path:
    # このモジュール（paths.py）のあるディレクトリを基準にする
    return Path(__file__).resolve().parent


def _output_dir() -> Path:
    return _base_dir() / "OUTPUT"


def _ensure_output_dir() -> Path:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _input_dir() -> Path:
    return _base_dir() / "INPUT"


def _ensure_input_dir() -> Path:
    inp = _input_dir()
    inp.mkdir(parents=True, exist_ok=True)
    return inp


# 起動時に標準フォルダを作成（従来挙動を維持）
INPUT_DIR = _ensure_input_dir()
OUTPUT_DIR = _ensure_output_dir()


def _resolve_kif_path(name: str) -> Path:
    """相対パス（ディレクトリ無し）の場合は OUTPUT/ に保存する。"""
    p = Path(name)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return _ensure_output_dir() / p.name


def _resolve_existing_kif_path(name: str) -> Path:
    """読み込み用。相対パスで見つからなければ OUTPUT/ も探す。"""
    p = Path(name)
    if p.exists():
        return p
    if not p.is_absolute() and p.parent == Path("."):
        q = _output_dir() / p.name
        if q.exists():
            return q
    return p
