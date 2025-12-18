#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
詰将棋探索（python-shogi）コア

Public API:
- solve_mate_tree
- count_solutions
- prune_tree_to_max_leaves
- enumerate_solution_paths
"""

from typing import Any, Dict, List, Optional, Tuple
import copy
import time

from models import SolveNode, SolveLimits

__all__ = [
    "solve_mate_tree",
    "count_solutions",
    "prune_tree_to_max_leaves",
    "enumerate_solution_paths",
]


def _is_attacker_turn(board: Any, attacker_turn: int) -> bool:
    return board.turn == attacker_turn


def solve_mate_tree(
    board: Any,
    ply_left: int,
    attacker_turn: int,
    check_only: bool = True,
    memo: Optional[Dict[Tuple[str, int], Optional[SolveNode]]] = None,
    stats: Optional[Dict[str, object]] = None,
    limits: Optional[SolveLimits] = None,
) -> Optional[SolveNode]:
    """
    Returns a SolveNode tree if attacker has a forced mate within ply_left plies, else None.
    The tree includes:
      - attacker nodes: all winning moves (OR)
      - defender nodes: all legal replies (AND), but only if all replies still lead to mate.
    """
    if memo is None:
        memo = {}
    if stats is None:
        stats = {"nodes": 0, "cutoff": False, "start": time.perf_counter(), "solutions": 0}
    if limits is None:
        limits = SolveLimits()

    stats["nodes"] = int(stats.get("nodes", 0)) + 1
    # time/node cutoff
    if stats["nodes"] > limits.max_nodes:
        stats["cutoff"] = True
        return None
    if limits.max_time_sec is not None:
        if (time.perf_counter() - float(stats.get("start", 0.0))) > float(limits.max_time_sec):
            stats["cutoff"] = True
            return None

    key = (board.sfen(), ply_left)
    if key in memo:
        return memo[key]

    if ply_left <= 0:
        memo[key] = None
        return None

    # If side to move is already checkmated, it means the previous move delivered mate.
    if board.is_checkmate():
        memo[key] = SolveNode(move=None, children=[])
        return memo[key]

    attacker_to_move = _is_attacker_turn(board, attacker_turn)

    legal = list(board.legal_moves)

    # Optional pruning: in tsume, attacker moves are usually checks only.
    if attacker_to_move and check_only:
        filtered = []
        for mv in legal:
            board.push(mv)
            ok = board.is_check()  # opponent is in check now
            board.pop()
            if ok:
                filtered.append(mv)
        legal = filtered

    if attacker_to_move:
        winning_children: List[SolveNode] = []
        for mv in legal:
            board.push(mv)
            if board.is_checkmate():
                stats["solutions"] = int(stats.get("solutions", 0)) + 1
                if stats["solutions"] > limits.max_solutions:
                    stats["cutoff"] = True
                    board.pop()
                    break
                winning_children.append(SolveNode(move=mv, children=[]))
                board.pop()
                continue
            child = solve_mate_tree(board, ply_left - 1, attacker_turn, check_only, memo, stats, limits)
            board.pop()
            if child is not None:
                winning_children.append(SolveNode(move=mv, children=child.children))
        node = SolveNode(move=None, children=winning_children) if winning_children else None
        memo[key] = node
        return node
    else:
        # Defender to move: attacker must survive all replies
        defender_children: List[SolveNode] = []
        for mv in legal:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                memo[key] = None
                return None
            child = solve_mate_tree(board, ply_left - 1, attacker_turn, check_only, memo, stats, limits)
            board.pop()
            if child is None:
                memo[key] = None
                return None
            defender_children.append(SolveNode(move=mv, children=child.children))
        node = SolveNode(move=None, children=defender_children)
        memo[key] = node
        return node


def count_solutions(tree: SolveNode) -> int:
    """Count distinct leaf mate lines in the tree."""
    def rec(node: SolveNode) -> int:
        if not node.children:
            return 1
        return sum(rec(ch) for ch in node.children)
    return rec(tree)


def prune_tree_to_max_leaves(tree: SolveNode, max_leaves: int) -> SolveNode:
    """Prune SolveNode tree (copy) so that total leaf lines <= max_leaves."""
    if max_leaves is None or max_leaves <= 0:
        return tree
    tree = copy.deepcopy(tree)

    def rec(node: SolveNode, remaining: List[int]) -> None:
        if remaining[0] <= 0:
            node.children = []
            return
        if not node.children:
            remaining[0] -= 1
            return
        new_children = []
        for ch in node.children:
            if remaining[0] <= 0:
                break
            rec(ch, remaining)
            new_children.append(ch)
        node.children = new_children

    rec(tree, [max_leaves])
    return tree


def enumerate_solution_paths(tree: SolveNode) -> List[List[object]]:
    """Return all move-sequences (shogi.Move list) from root to each leaf."""
    paths: List[List[object]] = []

    def rec(node: SolveNode, cur: List[object]) -> None:
        if node.move is not None:
            cur = cur + [node.move]
        if not node.children:
            paths.append(cur)
            return
        for ch in node.children:
            rec(ch, cur)

    rec(tree, [])
    return paths
