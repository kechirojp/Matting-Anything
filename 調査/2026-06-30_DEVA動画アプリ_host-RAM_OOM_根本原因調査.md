# DEVA 方式動画アプリ host-RAM OOM（`numpy._ArrayMemoryError`）根本原因調査

- 調査日: 2026-06-30
- 対象: `gradio_app_sam2_ben2_route_a_deva_for_Movie.py`（port 7863・DEVA 方式再構成）で、4K 動画に text_prompt="person" を実行すると約7分後に `numpy._core._exceptions._ArrayMemoryError` でクラッシュする事象
- 区分: 根本原因調査 ＋ 実装対処（ERR068・対処済み）
- 関連: `エラーログ/エラーログ_29.md`（原ログ）／`ERROR_LOG.md` ERR068／`REFERENCE.md` §5-7（DEVA 方式再構成パイプライン）

---

## 0. 結論（要約）

クラッシュの正体は **GPU メモリ（VRAM）ではなく host RAM（メインメモリ）の枯渇** である。

- `DevaSemiOnlineTracker` が **per-object logits（対象数 N × 高さ H × 幅 W の float32 配列）を、動画の全フレーム分・原寸のまま蓄積**していた。
- GroundingDINO が "person" を **19 box も過検出**（top_k=20 / box_threshold=0.25 既定）し、4K（4096×2160）解像度と重なって、

  $$
  19 \times 4096 \times 2160 \times 4\,\text{B} \approx 641\ \text{MiB / frame}
  $$

  これが **約190 frame 分積み上がり、合計 120 GB を超える** メモリ要求になって確保失敗した。

- ログ末尾の本当のエラーはこれ：

  ```text
  numpy._core._exceptions._ArrayMemoryError: Unable to allocate 641. MiB
  for an array with shape (19, 4096, 2160) and data type float32
  ```

- ログに約200件並ぶ Haystack の `WARNING`（後述）は **無害なノイズ**であり、原因ではない。**真のトレースはログの末尾**にある。

> **教訓**: Haystack はコンポーネント例外時にパイプライン状態を snapshot しようとして大量の警告を吐く。**ログは必ず最後まで読み、末尾の実トレースを原因とすること**（中間の警告に惑わされない）。

---

## 1. 観測されたエラー（エラーログ_29 の要点）

### 1-1. 真のエラー（ログ末尾）

```text
File ".../pipelines/components/video_model_components.py", line 753, in <...>
    per_object_logits[int(frame_index)] = np.stack(ordered, axis=0)
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 641. MiB
for an array with shape (19, 4096, 2160) and data type float32
```

- 直接の発生位置は `SAM2VideoPropagator` の `np.stack`（[video_model_components.py](../pipelines/components/video_model_components.py#L753)）。
- 呼び出し元は `DevaSemiOnlineTracker`（[deva_semi_online_tracker.py](../pipelines/components/deva_semi_online_tracker.py#L219) 付近）。
- shape `(19, 4096, 2160)` の `19` は **過検出された "person" の box 数**、`4096×2160` は 4K 解像度。

### 1-2. 無害なノイズ（ログに約200件並ぶ警告）

```text
WARNING haystack.core.pipeline.breakpoint - Failed to serialize ... nested functions ...
WARNING haystack.utils.base_serialization - Unsupported primitive type 'ndarray' ...
```

- これは Haystack 2.x が **コンポーネント例外時にパイプライン状態を snapshot** しようとし、`progress_callback`（クロージャ）や numpy 配列をシリアライズできずに出すもの。
- **OOM の副作用であって原因ではない**。エラーの本体は §1-1 の numpy トレース。

---

## 2. 根本原因の特定（コード上の確定事実）

### 2-1. tracker が全フレームの per-object logits を原寸蓄積していた

`DevaSemiOnlineTracker` は `global_per_object_logits`（`frame_idx → (N, H, W) float32`）を、追跡対象を含む全 source frame について保持する。

- N（対象数）は GroundingDINO の検出 box 数に比例する。
- 高解像度（4K）かつ N が大きい（過検出）と、1 frame あたりの配列が数百 MiB になり、フレーム数倍で host RAM を食い潰す。

### 2-2. per-object logits は「最終 α」ではなく「低周波の soft guard」

データの流れ：

```text
DevaSemiOnlineTracker.per_object_logits
  → OwnershipResolver（softmax-with-bg → 前景 soft guard）
  → BEN2 union ゲート（膨張 + ブラー）
```

- per-object logits は **そのまま最終アルファになるわけではなく**、膨張・ブラーされる**低周波のガイド（soft guard）**として使われる。
- **最終アルファは BEN2 が原寸で生成**する（per-object logits の解像度には依存しない）。
- したがって **per-object logits を縮小して蓄積しても実害は小さい**＝これが対処の設計根拠。

### 2-3. 過検出の背景

- GroundingDINO 既定の `top_k=20` / `box_threshold=0.25` で、群衆・複数人が映ると "person" が10〜20 box 検出され得る。
- これ自体は仕様内だが、4K と組み合わさると §2-1 のメモリ爆発を誘発する。

---

## 3. 対処（ERR068・実装済み・5箇所＋テスト）

config 駆動で per-object logits の解像度を上限化し、host RAM を **非有界 → 有界** に変える。原寸は `frame_hw` で持ち回り、下流で復元する。**全て後方互換**で、基底アプリ・propagator・`segment-anything/`・`samurai/` は無改変。

| # | ファイル | 変更点 |
|---|----------|--------|
| 1 | `config/route_a.toml` | `[deva] per_object_logits_max_side = 1024`（0=原寸=後方互換、低RAM環境は 512 推奨） |
| 2 | `pipelines/components/route_a_common.py` | `_DEFAULT_ROUTE_A_CONFIG["deva"]` 追加（既定 1024）。`load_route_a_config` のキーが `{alpha, blur_guide, composite, deva}` に |
| 3 | `pipelines/components/deva_semi_online_tracker.py` | `_downsample_per_object_logits`（`cv2.resize` INTER_AREA・空クリップ `(0,H,W)` も対応）。`run(per_object_logits_max_side=0)` 追加。>0 で縮小し原寸を `masks["frame_hw"]=(H,W)` に保持。overlay 用 `frame_masks` は原寸維持 |
| 4 | `gradio_app_sam2_ben2_route_a_deva_for_Movie.py` | `_deva_per_object_logits_max_side()` で config 値を読み込み、tracker へ注入。**union は config 値・per_object は 0（原寸）** |
| 5 | `pipelines/components/ownership_resolver.py` | `frame_hw` があれば前景 `frame_masks` を `cv2.resize` INTER_LINEAR で原寸へ復元（不在時は従来どおり縮小なし＝後方互換） |

### 3-1. メモリ削減効果

| 条件 | 1 frame あたり | 削減率 |
|------|----------------|--------|
| 修正前（原寸 4096×2160） | 約 641 MiB | — |
| 修正後（長辺 1024 上限） | 約 45 MiB | 約 1/14 |

- 約190 frame で 120 GB 超 → 約 8.5 GB 程度へ。**非有界 → 有界** に変化。

### 3-2. 不変条件（follow-up として記録）

- 「`max_side>0` ⟹ union モード」という不変条件は **アプリ層（per_object は 0 を渡す）でのみ担保**している。
- 将来 per_object に `max_side>0` を渡す呼び出しが現れた場合、`composite_alpha_by_ownership` が shape 不一致で `ValueError`（silent ではない）を送出する＝安全側に倒れる。UI から到達不能のため現時点では非ブロッカーの follow-up とする。

---

## 4. 検証

- **非 integration 347 passed / 3 deselected**（+5 新規テスト・回帰なし）
  - `test_deva_semi_online_tracker.py` +3（原寸維持で frame_hw なし／縮小＋frame_hw 付与／空クリップ縮小）
  - `test_ownership_resolver.py` +2（frame_hw ありで原寸復元／なしで従来解像度維持）
  - `test_route_a_common.py` 更新（`deva` キー／型）
- `get_errors` = 0、DEVA アプリ `--help` smoke 成功。
- サブエージェントレビュー：**マージ可（critical なし）**。
- **ERR035 非該当**（UI レンダリング変更ではなくバックエンドのメモリ修正）→ Playwright 実機検証は不要。RED→GREEN は単体テストで実施。

---

## 5. 教訓・再発防止

1. **ログは末尾まで読む**。Haystack の snapshot-on-exception 警告は大量に出るが無害なノイズで、真の原因は末尾のトレースにある。
2. **フレーム蓄積型の配列は host RAM 量を見積もる**：`N × H × W × 4B × frame数`。N が過検出で膨らみ得る／解像度が 4K の場合は特に危険。
3. **soft guard（低周波ガイド）は解像度を落として良い**。最終 α は BEN2 が原寸生成するため、ガイドの縮小は品質にほぼ影響しない。
4. 解像度・上限値は **config 駆動**（`config/route_a.toml [deva]`）でハードコードしない。原寸は `frame_hw` 契約で持ち回り、下流（OwnershipResolver）で復元する。
