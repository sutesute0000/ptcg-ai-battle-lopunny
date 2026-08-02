# PTCG AI Battle Challenge — Mega Lopunny ex agent

[Pokémon Trading Card Game AI Battle Challenge](https://ptcg-abc.pokemon.co.jp/)
(Kaggle × The Pokémon Company × 松尾研究所 × HEROZ) シミュレーション部門の
参加コード。`agent(obs) -> list[int]` を実装し、自動対戦ラダーで戦う。

- シミュレーション部門 締切: 2026-08-17 08:59 JST
- ストラテジー部門(レポート)締切: 2026-09-14

## 方針

**強いデッキを1つ選び、そのデッキが生み出す勝ちパターンを実データから特定し、
それを押し付けるエージェントを作る。** 相手の手を読む処理は持たない。

採用デッキは **Mega Lopunny ex**(2026-08-01 のトップ層で勝率61.2%と最高)。
勝ち筋・操縦ルール・検証結果は [`docs/deck_strategy.md`](docs/deck_strategy.md)、
全体計画は [`docs/plan.md`](docs/plan.md)。

## 構成

```
agents/lopunny/     エージェント本体 (main.py + deck.csv + build_submission.sh)
docs/               計画とデッキ戦略の記録
tools/              分析・検証ツール
```

| ツール | 役割 |
|---|---|
| `divergence.py` | トップパイロットの実対局を自エージェントでリプレイし、選択の不一致をカード名にデコードする。操縦ルールはここから導出する |
| `ctx_probe.py` | 特定の判断が「どの盤面特徴で」分岐しているかをクロス集計する |
| `autopsy.py` | 自分のラダー対局のリプレイを取得し、相手アーキタイプ別の勝率を出す |
| `gauntlet.py` | 実メタのデッキ相手にシェア重み付き勝率を測る(版の優劣判定) |
| `make_opponents.py` | エピソードデータから実際に勝っているデッキリストを抽出し、対戦相手を生成 |
| `eval_local.py` | 2つのエージェントディレクトリを直接対戦させる |
| `meta_analyze.py`(参照) | アーキタイプ分布・勝率・相性表 |

## セットアップ

公式のエンジンとカードデータは再配布しないので、Kaggle から取得する。

```bash
python3 -m venv .venv
.venv/bin/pip install --no-deps kaggle-environments==1.30.1
.venv/bin/pip install kaggle numpy pandas jsonschema requests Flask pydantic

# 公式スターター(エンジン cg/ を含む)
.venv/bin/kaggle competitions download pokemon-tcg-ai-battle -p docs/official/
(cd docs/official && unzip -q pokemon-tcg-ai-battle.zip -d starter)

# cabt 環境は libcg.so 決め打ちで arm64 mac では動かないため、
# platform 判定を持つ公式スターターの cg/ で上書きする
cp docs/official/starter/sample_submission/sample_submission/cg/* \
   .venv/lib/python3.*/site-packages/kaggle_environments/envs/cabt/cg/
```

エージェントを動かすには `agents/lopunny/cg/`(エンジン)が必要:

```bash
cp -R docs/official/starter/sample_submission/sample_submission/cg agents/lopunny/cg
```

## ビルドと提出

```bash
bash agents/lopunny/build_submission.sh
.venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f agents/lopunny/submission.tar.gz -m "message"
```

提出は1日5回、直近2つがラダーで稼働する。提出には Kaggle の
Identity Verification(電話番号認証とは別)が必要。

## 検証の作法

このリポジトリで繰り返し痛い目を見た点なので、変更時は必ず守る。

1. **比較したい2択が両方とも選べた場面に絞ってから数える。** 条件付けを誤ると
   もっともらしい誤結論が出る(「倒された後は捨て駒を出す」は、準備済み
   ミミロップが選べなかった場面を混ぜた集計による誤りだった)。
2. **行動の「回数」は優先「順位」ではない。** 多く打たれている札は単に
   デッキに多く入っているだけかもしれない。
3. **1相手40戦では判定できない。** 同一条件で 92%→82%、70%→55% と振れた。
   版の優劣を決めるなら最低120戦/相手。
4. **相手ボットの質が結果を支配する。** 汎用ポリシーの Alakazam 相手は55%、
   本格パイロット相手は33%、実ラダーの相性は35%。汎用ボットの数字は
   上限値であって予測値ではない。
5. **ミラー戦(自分同士)はラダー強度を予測しない。**

## ライセンス / 帰属

The Pokémon Company / Kaggle とは無関係の個人プロジェクト。ポケモンおよび
カード名は各権利者の商標。公式のエンジン・カードデータ・エピソードデータは
本リポジトリに含まない。
