# 将棋 KIF 入力 TUI

Textual を用いた、将棋局面編集・指し手入力用の TUI アプリケーションです。  
KIF 作成前の「入力・確認・修正」を快適に行うことを目的としています。

---

## 特徴

- Textual ベースの TUI
- vim 風 NORMAL / INPUT モード
- 盤面カーソル操作
- 駒配置（p コマンド）
- 数字指し手入力（例: 7776）
- undo / clear / start / sfen / kif
- **自動 smoke テスト基盤（再現可能）**

---
## 設計メモ

- 状態管理は `ShogiPosition` に集約
- TUI は「状態を表示するだけ」に近づける
- 操作の正否は **必ずログに出す**
- smoke test は「仕様の生きたドキュメント」

---

## 起動方法

```bash
python tui_app.py
