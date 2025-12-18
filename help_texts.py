#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# ----------------- Help -----------------

HELP_MAIN = """\
============================================================
  詰将棋用 KIF 入力 CUIツール（v4）
============================================================

【最短の流れ】
  1) 局面作成: show → p / h / turn → start
  2) 手順入力: 7776 / 22331 / 076(打ち)
  3) 保存:     s out.kif

【発展：自動で全詰み筋を探索（python-shogi必要）】
  solve 9 out.kif
    - 開始局面（startで確定）から、9手詰（=9プライ）以内の
      「先手の強制詰み筋」を探索して、分岐付きKIFに出力します。
    - 先手の手は基本「王手になる手」に限定（詰将棋向け高速化）

【コマンド】
  show                 : 盤面表示
  p 55                 : 駒配置（メニュー）
  p 55 v+R / p 55 .     : 駒配置（直指定 / 消す）
  h b P 2 / h w R 1     : 先手/後手の持ち駒（※後手は出力時に自動残駒化も可能）
  turn b / turn w      : 手番
  start                : 開始局面を確定（ここから手順入力）
  end                  : 終局を選択（詰み/投了） ※省略可（詰み自動判定あり）
  7776 / 22331         : 移動（4桁）/ 成り（末尾1）
  076                  : 打ち（0+先2桁 → 候補選択）
  u                    : 1手戻す（手順のみ）
  s file.kif           : 保存（cp932）
  solve 9 out.kif      : 全詰み筋探索→分岐KIF出力（python-shogi必要）
  possave file.pos     : 開始局面のみ保存（すぐstart可）
  posload file.pos     : 開始局面を読み込み（start済み）
  preview file.kif 80  : 保存済みKIFのプレビュー表示（先頭80行）
  batch [INPUT] 9      : 既存KIF群を一括処理（変化→分割 / 無変化→解探索→分割）。INPUT省略可
  help [topic]         : サブヘルプ（topic: setup / p / h / move / drop / save / solve）
  q                    : 終了
============================================================
"""

HELP_SOLVE = """\
[help solve]
solve は開始局面から「指定手数以内の強制詰み筋」を探索して、分岐KIFに出します。

使い方:
  start
  solve 9 out.kif

ポイント:
  ・手数は 1,3,5,7,9 のような奇数を想定（詰将棋）
  ・指定手数を超えたら「未解決」として打ち切ります
  ・先手側の候補手は「王手になる手」に限定して高速化します
  ・余詰があると分岐が増えます（KIFの変化として出ます）

注意:
  ・python-shogi が必要です（pip install python-shogi）
"""


HELP_SETUP = """\
[help setup]
局面作成（開始局面の作り方）

基本:
  show                 盤面確認
  p 55                 マス55に駒を置く（候補メニュー）
  p 55 v+R              直指定で置く（例: 後手の竜）
  p 55 .               駒を消す
  h b G 1              先手の持駒（例: 金1枚）
  turn b / turn w      手番（先手= b / 後手= w）
  start                この局面を開始局面として確定（ここから手順入力・solve可能）

駒トークンの例:
  P L N S G B R K       先手の歩/香/桂/銀/金/角/飛/玉
  vP vK                 後手の歩/玉
  +P +R                 成り（と/竜）※後手なら v+P のように書きます
  .                      空（消す）

よくある流れ（詰将棋）:
  p 42 vK
  p 44 P
  p 34 G
  h b G 1
  turn b
  start
"""

HELP_P = """\
[help p]
駒配置コマンド

形式:
  p 55           : メニューから選んで配置（おすすめ）
  p 55 v+R       : 直指定で配置（後手の竜）
  p 55 .         : 駒を消す

メニュー操作:
  - 番号で選択
  - 0 を選ぶと「直前に置いた駒」を再利用できます（高速配置用）
"""

HELP_H = """\
[help h]
持ち駒編集コマンド（局面作成時）

形式:
  h b P 2        : 先手(▲)の歩を2枚
  h w R 1        : 後手(△)の飛を1枚（※ツール内部では保持しますが、KIF出力時は“残駒自動計算”が主）
  h b G 0        : 0にするとその駒種を削除

注意:
  - K は持ち駒にできません
  - 枚数は 0以上の整数
"""

HELP_MOVE = """\
[help move]
手順入力（移動/成り）

形式:
  7776           : 7七→7六
  22331          : 2二→3三 成り（末尾の1が「成り」）

補足:
  - start 後に入力してください
  - “同” 表記は直前の着手先と同じマスのとき自動になります
"""

HELP_DROP = """\
[help drop]
手順入力（打ち）

形式:
  076            : 「0 + 行先2桁」→ そのマスに打てる駒候補を表示 → 番号で選択

例:
  055            : 5五に打つ（候補から選ぶ）

注意:
  - そのマスに駒がある場合は打てません
  - 二歩などは簡易チェックしています（python-shogi導入時はより厳密）
"""

HELP_SAVE = """\
[help save]
保存と読み込み/プレビュー

KIF保存:
  s out.kif

開始局面だけ保存（すぐstartできる）:
  possave mypos.pos
  posload mypos.pos   : 読み込み後は start 済み扱いになります

KIFプレビュー:
  preview out.kif
  preview out.kif 80  : 先頭80行だけ表示
"""

HELP_MAP = {
    "setup": HELP_SETUP,
    "p": HELP_P,
    "h": HELP_H,
    "move": HELP_MOVE,
    "drop": HELP_DROP,
    "save": HELP_SAVE,
    "solve": HELP_SOLVE,
}

# 「公開API」を明示（任意だけど気持ちいい）
__all__ = [
    "HELP_MAIN",
    "HELP_SOLVE",
    "HELP_SETUP",
    "HELP_P",
    "HELP_H",
    "HELP_MOVE",
    "HELP_DROP",
    "HELP_SAVE",
    "HELP_MAP",
]