#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any


@dataclass
class Piece:
    color: str  # "B" (sente) or "W" (gote)
    kind: str   # "P,L,N,S,G,B,R,K"
    prom: bool = False


@dataclass
class Move:
    is_drop: bool
    kind: str
    from_sq: Optional[Tuple[int, int]]
    to_sq: Tuple[int, int]
    promote: bool
    same_as_prev: bool


@dataclass
class SolveNode:
    move: Optional[Any]  # shogi.Move or None for root
    children: List["SolveNode"]


@dataclass
class SolveLimits:
    max_nodes: int = 50000
    max_time_sec: float = 5.0
    max_solutions: int = 300
