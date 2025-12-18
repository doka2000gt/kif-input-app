# kif-tsume-cui-solver

詰将棋向けの CUI ツールです。

- KIF形式の詰将棋を読み込み
- python-shogi を用いた詰探索
- 解答筋を ANKIF 互換 KIF として出力
- 既存KIFの分岐（変化）を分割・整理

## Requirements

- Python 3.10+
- python-shogi（任意・あると解探索が有効）

```bash
pip install python-shogi


---

### ③ ディレクトリ構成（← これが一番価値あります）

```md
## Project Structure

kif_tsume_cui_solver.py # CLI / エントリポイント
constants.py # 定数
models.py # データモデル
position.py # ShogiPosition（局面管理）
helpers.py # 汎用ヘルパー
kif_format.py # KIF生成
kif_parser.py # KIF解析
solver_core.py # 詰探索ロジック
batch_runner.py # batch処理
sfen.py # SFEN生成
help_texts.py # HELP文言
tests/ # pytest
INPUT/ # 入力KIF
OUTPUT/ # 出力KIF

## Usage

### 単体起動
```bash
python kif_tsume_cui_solver.py

batch INPUT 9
- INPUT 内の KIF を 9手詰で一括処理
- 解答筋ごとに OUTPUT に KIF を生成


---

### ⑤ テスト

```md
## Tests

```bash
pytest


---

### ⑥ 設計方針（← 書いておくと未来の自分が泣いて喜ぶ）

```md
## Design Notes

- 責務ごとにモジュールを分離
- solver / parser / formatter / position を明確に分割
- pytest により「安心して壊せる」構成を維持
- 詰将棋用途に特化し、最小合法性のみを実装
