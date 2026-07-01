# 人物→transparent-background / 物体→BEN2 クラス別振り分け＋最終α合成 計画

- 作成日: 2026-06-30
- 区分: 後続計画（**計画書1「毎フレーム検出ByteTrack追跡_A案ルートA」が成功したら着手**）
- 前提: 計画書1で追跡（検出→ByteTrack→ID選別）が動き、track ごとに **ラベル（人物/物体）** を引き継げていること
- 関連: `計画書/2026-06-30_毎フレーム検出ByteTrack追跡_A案ルートA_実装計画.md`

---

## 1. 目的

被写体の種類に応じて **最適なマッティングモデルへ振り分け**、最後に α を合成して 1 枚の RGBA に統合する。

- **人物** → `transparent-background`（人物・ポートレート特化。毛先のソフト α が得意）
- **物体** → `BEN2`（ルートA: ブラー誘導。一般物体の matting）
- **最終合成** → 2 系統の α を per-frame で合成（over / max）し、単一 RGBA を書き出す

計画書1で track に乗せた **ラベル（GroundingDINO の phrase / COCO クラス）** が、そのまま振り分けの判定材料になる（疎結合の利得）。

---

## 2. フロー図（ASCII）

### 2.1 クラス別振り分け＋最終合成

```
            計画書1 の追跡結果（track ごとに box＋label）
                              │
                              ▼
                   ┌────────────────────┐
                   │  ClassRouter        │   label で振り分け
                   │  人物? / 物体?       │
                   └───────┬───────┬─────┘
              person tracks│       │object tracks
                           ▼       ▼
        ┌──────────────────────┐  ┌──────────────────────────┐
        │ TransparentBG        │  │ BEN2RouteAVideoExtractor  │
        │ VideoExtractor       │  │ （ブラー誘導 → 再α化）     │
        │ 人物 α（毛先ソフト）  │  │ 物体 α                    │
        └──────────┬───────────┘  └────────────┬─────────────┘
                   │ α_person{f:(H,W)}          │ α_object{f:(H,W)}
                   └─────────────┬──────────────┘
                                 ▼
                   ┌────────────────────────────┐
                   │  AlphaCompositor            │   per-frame 合成
                   │  α_out = compose(           │   （over / max / 優先順）
                   │     α_person, α_object )     │
                   │  → RGBA 合成                 │
                   └──────────────┬──────────────┘
                                  │ matte
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────────────┐
              │VideoWriter│ │FrameSequenceWriter│ …
              └──────────┘ └──────────────────┘
                    │
                    ▼
              統合 RGBA 動画 / 連番PNG
```

### 2.2 α 合成の考え方

```
  人物 α（transparent-background）   物体 α（BEN2）
        ███▓▓░                          ░▒██
        ██████  ← 毛先ソフト             ▒███  ← 物体エッジ
        ▓▓░                              ██
              ╲                        ╱
               ╲   AlphaCompositor    ╱
                ▼  α_out = max/over  ▼
                   ███▓▓░▒██
                   ████████   ← 両者を保持して 1 枚に統合
                   ▓▓░ ██

  ※ 重なり領域は前後関係（優先順 or α 比較）で決める。
  ※ MVP は単純 max 合成から開始、必要なら over（前後）に拡張。
```

---

## 3. 新規 / 再利用 Component

| 役割 | Component | 区分 | 備考 |
|------|-----------|------|------|
| 振り分け | `ClassRouter`（新規） | 純関数寄り | label→{person, object} に track を分割。判定語彙は `config/*.toml` 化 |
| 人物 α | `TransparentBGVideoExtractor`（既存） | 再利用 | `video_model_components.py` |
| 物体 α | `BEN2RouteAVideoExtractor`（既存） | 再利用 | ルートA |
| 合成 | `AlphaCompositor`（新規） | 純関数寄り | 2 系統の α を per-frame 合成し RGBA 化 |
| 書出 | `VideoWriter` / `FrameSequenceWriter`（既存） | 再利用 | matte 契約に整合 |

- I/O 契約は計画書1の masks / matte 契約に揃え、socket 型を固定（差し替え可能性を維持）。
- `ClassRouter` / `AlphaCompositor` はモデル非依存の純関数 → pytest 容易（先にテスト）。

---

## 4. TODO

### Phase 0 — 前提確認（計画書1 完了後）
- [ ] 計画書1 の track に **ラベル**が保持されていることを確認（人物/物体の判定が可能か）
- [ ] `transparent-background` の重み（`checkpoints/transparent_BG/`）存在確認
- [ ] 人物判定の語彙ルールを設計（`person`/`man`/`woman`… / COCO `person`）→ `config` 化方針決定

### Phase 1 — 新規 Component（RED→GREEN）
- [ ] `ClassRouter` テスト（label→person/object 分割、未知ラベルの既定）→ 実装
- [ ] `AlphaCompositor` テスト（max / over 合成、空入力、片系統のみ）→ 実装
- [ ] 合成出力が VideoWriter の matte 契約に一致することを確認

### Phase 2 — Pipeline 配線
- [ ] 振り分け→2 系統 α→合成→書出 を Haystack で connect
- [ ] 人物のみ / 物体のみ / 混在 の 3 ケースで配線が破綻しないこと

### Phase 3 — Gradio 統合
- [ ] 計画書1 のアプリに「クラス別振り分け＋合成」モードを追加（or 新タブ）
- [ ] 合成方式（max / over / 優先順）を UI で選択可能に

### Phase 4 — 検証
- [ ] `.venv\Scripts\python.exe -m pytest -m "not integration" -q`
- [ ] `--help` 起動 smoke
- [ ] Playwright 実行時検証（UI 配線、ERR035 準拠）
- [ ] サブエージェントレビュー（socket 型 / 合成境界の正しさ / `weights_only=True`）

### 完了処理
- [ ] `WHITEBOARD.md` / `ERROR_LOG.md` 更新

---

## 5. 留意・リスク

- 人物判定はラベル依存。GroundingDINO（自由語彙）と RF-DETR（COCO）で語彙が異なるため、判定ルールを `config` で吸収する。
- 重なり領域の前後関係（occlusion）は MVP では単純化（max）。厳密な前後は計画書1 のスコープ外（奥行き判断）と連動するため、別タスク扱い。
- transparent-background と BEN2 で α の性質（しきい/ソフト度）が異なる → 合成前に正規化方針を決める。
- `torch.load(..., weights_only=True)` 厳守。`try/except: pass` 禁止。既存正常系は変更しない。
