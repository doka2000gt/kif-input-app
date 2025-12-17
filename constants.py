#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------- constants -----------------

FW_DIGITS = {str(i): ch for i, ch in enumerate("０１２３４５６７８９")}
RANK_KANJI = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}

PIECE_JP = {"P":"歩","L":"香","N":"桂","S":"銀","G":"金","B":"角","R":"飛","K":"玉"}
PROMOTED_JP = {"P":"と","L":"成香","N":"成桂","S":"成銀","B":"馬","R":"竜"}
PROMOTABLE = set(["P","L","N","S","B","R"])

# Total piece counts in standard shogi (excluding kings in "hands" output)
TOTAL_COUNTS = {"R":2,"B":2,"G":4,"S":4,"N":4,"L":4,"P":18,"K":2}

# piyo-like board display tokens
KIND_TO_PYO = {
    ("P",False):"歩", ("L",False):"香", ("N",False):"桂", ("S",False):"銀", ("G",False):"金",
    ("B",False):"角", ("R",False):"飛", ("K",False):"玉",
    ("P",True):"と", ("L",True):"成香", ("N",True):"成桂", ("S",True):"成銀",
    ("B",True):"馬", ("R",True):"竜",
}
