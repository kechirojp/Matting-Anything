# DESIGN.md — LayerMaker (画像レイヤー分解アプリ)

> **このドキュメントは AI コードエージェント向けのデザインシステム仕様書です。**
> UI コンポーネント実装・スタイル決定・コードレビュー時、すべてのフロントエンド変更はこの仕様に従ってください。
> 値（hex / px / 数値）は唯一の正典であり、ハードコードを避けるため `src/frontend/src/styles/tokens.css`（Phase 2 で生成予定）に同期されます。

---

## 0. Design Philosophy

**"Workshop, not showroom."** — 本アプリは職人の作業台であり、ショールームではない。

- ユーザーの**画像が常に主役**であり、UI は意図的に「目立たない」設計を取る
- 参照系譜：**Adobe Photoshop / Affinity Photo / Lightroom / Linear / Warp / Raycast / MUJI / Apple Japan**
- 拒絶系譜：Stripe（紫グラデ）／Cohere（カラフル SaaS）／Lovable（プレイフル AI）／一般的な SaaS マーケティングサイト
- ターゲット：**プロの VTuber クリエイター・イラストレーター**。日本市場主、12 言語対応
- 価格帯メッセージ：$9.9 買い切り全部入り → 「高級感」ではなく「**実直で速い職人道具**」感を出す

5 つの形容詞でこの製品を表すなら：
**Quiet, Precise, Dense, Honest, Japanese-aware**

---

## 1. Visual Theme & Atmosphere

| 項目 | 値 / 方針 |
|---|---|
| **メタファー** | 暗室の作業台。中性的なキャンバスにユーザー画像のみが発色する |
| **主軸テーマ** | Light（主）／ Dark（従）。`prefers-color-scheme` に追従（MVP は system のみ） |
| **キャンバス哲学** | ライトは「無漂白の和紙」、ダークは「現像トレイ」。**純白 `#FFFFFF` と純黒 `#000000` は使わない** |
| **密度** | プロツール密度。Linear / Photoshop 寄り。Notion より明確に密 |
| **質感** | フラット。ガラス・グラデ・ネオン・グロウは禁止 |
| **境界処理** | 1px ヘアライン主体。シャドウは控えめ（最大 elevation-3 まで） |
| **動き** | 機能的トランジションのみ（120〜180ms / `cubic-bezier(0.2, 0, 0, 1)`）。装飾アニメーション禁止 |

---

## 2. Color Palette & Roles

### 2.1 Light Theme（主）

| Role | Token | Hex | 用途 |
|---|---|---|---|
| Canvas | `--canvas` | `#F4F2EC` | アプリ最背面（無漂白の和紙） |
| Surface-1 | `--surface-1` | `#F8F5EE` | パネル・サイドバー背景 |
| Surface-2 | `--surface-2` | `#FCFAF4` | カード・モーダル前景（暖色寄りの紙面・純白不使用） |
| Border | `--border` | `#D8D3C7` | 1px ヘアライン |
| Border-strong | `--border-strong` | `#9E978A` | フォーカス枠・区切り強 |
| Text-primary | `--text-1` | `#1C1A16` | 本文（純黒不使用） |
| Text-secondary | `--text-2` | `#5E5A50` | 補助・キャプション |
| Text-tertiary | `--text-3` | `#8A8478` | 非活性・プレースホルダ（border-strong と値を分離） |
| Accent | `--accent` | `#EAE7DE` | アクセント面（ペーパーホワイト） |
| Accent-fg | `--accent-fg` | `#1C1A16` | アクセント上の文字 |
| Accent-edge | `--accent-edge` | `#1C1A16` | アクセントの 1px 縁取り（識別性確保） |
| Success | `--success` | `#4F6A43` | 完了・保存成功（Canvas 比 ~5.5:1 / AA 合格） |
| Warning | `--warning` | `#8C5E1F` | VRAM 警告・モデル未取得（Canvas 比 ~4.9:1 / AA 合格） |
| Error | `--error` | `#9C3B2E` | 失敗・エラーログ（Canvas 比 ~5.4:1 / AA 合格） |
| Focus | `--focus` | `#1C1A16` | キーボードフォーカス（2px outline + 2px offset） |

### 2.2 Dark Theme（従）

| Role | Token | Hex | 用途 |
|---|---|---|---|
| Canvas | `--canvas` | `#16140F` | アプリ最背面（現像トレイ） |
| Surface-1 | `--surface-1` | `#1E1B15` | パネル・サイドバー背景 |
| Surface-2 | `--surface-2` | `#26221B` | カード・モーダル前景 |
| Border | `--border` | `#37322A` | 1px ヘアライン |
| Border-strong | `--border-strong` | `#5E5A50` | フォーカス枠・区切り強 |
| Text-primary | `--text-1` | `#EAE7DE` | 本文（純白不使用） |
| Text-secondary | `--text-2` | `#A8A294` | 補助・キャプション |
| Text-tertiary | `--text-3` | `#6E695E` | 非活性・プレースホルダ |
| Accent | `--accent` | `#EAE7DE` | アクセント面（ペーパーホワイト） |
| Accent-fg | `--accent-fg` | `#16140F` | アクセント上の文字 |
| Accent-edge | `--accent-edge` | `#EAE7DE` | （ダークでは縁取り不要・面で識別） |
| Success | `--success` | `#7A9966` | 完了・保存成功 |
| Warning | `--warning` | `#C99A56` | VRAM 警告・モデル未取得 |
| Error | `--error` | `#C56656` | 失敗・エラーログ |
| Focus | `--focus` | `#EAE7DE` | キーボードフォーカス（2px outline + 2px offset） |

### 2.3 Color Rules

- **アクセントは 1 系統のみ**（ペーパーホワイト `#EAE7DE`）。グラデーション・複数アクセントの追加禁止
- **コントラスト**：本文と背景は WCAG AA（4.5:1）以上、UI コンポーネントは 3:1 以上を保証
- **Text-tertiary（`--text-3`）は placeholder / disabled 専用**。本文・キャプションには使用しない（プレースホルダは WCAG 例外的に 3:1 を許容）
- **セマンティックカラーは状態通知専用**。装飾目的（タグの色分け等）に Success / Warning / Error を流用しない
- **Success / Warning / Error をテキスト着色として使う場合は Canvas / Surface-1 / Surface-2 上で AA を満たす値のみ採用**（§2.1 / §2.2 の値はこれを保証）
- **画像が乗る領域は常に Canvas / Surface-1**。Surface-2（前景カード）の上に画像を表示しない（色被りで作品を歪めない）

---

## 3. Typography Rules

### 3.1 和文（Japanese）

- 主フォント：**Noto Sans JP**（400 / 500 / 700）
- ウェイト運用：本文 400、見出し 500、強調のみ 700。**900 は使用禁止**
- `font-feature-settings`：本文 `"palt" off`（プロポーショナルメトリクスは見出しのみ）

### 3.2 欧文（Latin / Numeric）

- 主フォント：**Inter**（400 / 500 / 600）+ `tabular-nums`
- 数値（進捗 % ・寸法 px ・解像度）は必ず `font-variant-numeric: tabular-nums` を適用
- **強調ウェイト**：和文 700 と欧文 600 を対応関係として揃える（混植時の視覚重み一致）

### 3.3 font-family Stack（多言語・必ず 3 段以上のフォールバック）

12 言語対応（`SPECIFICATION.md §6.1`）に向け、CJK・タイ語の字形混在を防ぐため `:lang()` で切替える。

```css
--font-sans:
  "Inter",
  "Noto Sans JP",
  -apple-system, BlinkMacSystemFont,
  "Hiragino Sans", "Yu Gothic UI", "Meiryo",
  system-ui, sans-serif;

--font-mono:
  "JetBrains Mono",
  "SF Mono", Menlo, Consolas,
  "Noto Sans Mono CJK JP",
  monospace;
```

言語別オーバーライド（CJK の字形差・タイ語の字形を保証）：

```css
:lang(zh-Hans) { font-family: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif; }
:lang(zh-Hant) { font-family: "Inter", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif; }
:lang(ko)      { font-family: "Inter", "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif; }
:lang(th)      { font-family: "Inter", "Noto Sans Thai", "Sarabun", system-ui, sans-serif; }
```

### 3.4 階層（Hierarchy）

| 用途 | size / line-height / weight | letter-spacing |
|---|---|---|
| Display（ウィザードのみ） | 28px / 1.4 / 500 | -0.01em |
| H1（画面タイトル） | 20px / 1.5 / 500 | 0 |
| H2（セクション） | 16px / 1.5 / 500 | 0 |
| H3（サブセクション） | 14px / 1.5 / 500 | 0 |
| Body | 14px / 1.7 / 400 | 0.01em（和文） |
| Body-sm（補助） | 13px / 1.6 / 400 | 0.01em |
| Caption（メタ情報） | 12px / 1.6 / 400 | 0.02em |
| Code / Numeric | 13px / 1.5 / 400（mono + tabular-nums） | 0 |

### 3.5 行間・字間（和文最重要）

- **和文 line-height は 1.6 を下限**（Body は 1.7 推奨）
- **letter-spacing**：和文 `0.01em〜0.02em`、欧文 `0`、見出し `-0.01em` まで
- 1 行の最適長：**和文 30〜40 字、混植 40〜60 字**。リスト幅もこれに準ずる

### 3.6 禁則処理

- 既定（CJK 系）：`word-break: normal; overflow-wrap: anywhere; line-break: strict;`
- 行頭禁則：`、 。 ） ］ ｝ 」 』 】 !  ?`
- 約物の二重出現（`、、` `。。`）はバリデーションで弾く
- **タイ語例外**：`:lang(th) { line-break: auto; word-break: normal; overflow-wrap: normal; }`（タイ語は単語境界を持たないため `strict` 禁則は不可）
- **欧文例外**：`:lang(en), :lang(es), :lang(fr), :lang(de), :lang(pt), :lang(ru) { word-break: normal; line-break: auto; }`

### 3.7 OpenType Features

- 本文：`"palt" off, "kern" on, "calt" on`
- 見出し：`"palt" on, "kern" on`（プロポーショナル詰めは見出しのみ）
- 数字：常に `"tnum" on`（テーブル・進捗・寸法）

### 3.8 縦書き（N/A）

- 本アプリは横書きのみ。`writing-mode: vertical-rl` は将来も使用しない

---

## 4. Component Stylings

### 4.1 共通ルール

- **角丸**：`--radius-sm: 2px` / `--radius-md: 4px` / `--radius-lg: 6px`。**8px 以上の角丸は禁止**（職人道具の精密感）
- **境界**：1px solid `--border`。Hover で `--border-strong` に遷移（120ms）
- **影**：原則使わない。モーダル・ドロップダウンのみ最小限（§6 参照）
- **タッチターゲット**：本アプリはデスクトップ専用のため最小 24×24px（マウス前提）

### 4.2 Button

| Variant | 用途 | 仕様 |
|---|---|---|
| Primary（Light） | 「分解する」など主要アクション | `bg: --accent / fg: --accent-fg / 1px solid --accent-edge` |
| Primary（Dark） | 同上 | `bg: transparent / fg: --accent / 1px solid --accent`（アウトライン化。ダークでアクセント面ベタは「画像が主役」原則と衝突するため面ではなく縁で強調） |
| Secondary | キャンセル・サブ | `bg: --surface-2 / fg: --text-1 / 1px solid --border` |
| Ghost | アイコンボタン・メニュー項目 | `bg: transparent / fg: --text-1`、Hover で `--surface-2` |
| Destructive | 削除・リセット | `bg: transparent / fg: --error / 1px solid --error`、Hover で `bg: --error / fg: --text-1`（ライトは `--surface-2` 相当の白寄り、ダークは `--text-1`＝ペーパーホワイトで AA 確保） |

- Padding：`8px 14px`（md） / `6px 10px`（sm）
- Disabled：`opacity: 0.4; cursor: not-allowed;`（色変更ではなく不透明度で示す）

### 4.3 Input / Textarea / Select

- 高さ 32px、padding `6px 10px`、`bg: --surface-2`、`border: 1px solid --border`
- Focus：`outline: 2px solid --focus; outline-offset: 2px; border-color: --border-strong`
- Placeholder：`color: --text-3`

### 4.4 Slider（LoRA 強度・閾値調整に使用）

- Track 高さ 2px、Thumb 14×14px の正方形（角丸 2px、職人感のため**円ではなく角形**）
- Thumb：`bg: --accent / 1px solid --accent-edge`
- ステップ刻みは 0.05 単位で tabular-nums 表示

### 4.5 Checkbox / Radio

- 16×16px、角丸 `--radius-sm`
- チェック時：`bg: --accent / fg: --accent-fg`、チェックマークは Lucide `check`（線 1.5px）

### 4.6 Card / Panel

- `bg: --surface-1`（パネル）または `--surface-2`（前景カード）
- 1px `--border`、角丸 `--radius-md`
- 内部 padding `12px 16px`
- **影は使わない**（パネルはヘアラインで階層を表現）

### 4.7 Modal / Dialog

- バックドロップ：`rgba(22, 20, 15, 0.5)`（ダーク値ベースのため両テーマ共通）
- 本体：`bg: --surface-2 / 1px solid --border-strong`、角丸 `--radius-lg`、影 `--elevation-3`
- 最大幅 480px（標準）／ 720px（ウィザード）

### 4.8 Toast

- 右下固定、幅 320px
- 1px `--border`、角丸 `--radius-md`、影 `--elevation-2`
- Success / Warning / Error の左 2px ストライプで識別（背景色変更は禁止）

### 4.9 ProgressBar（分解処理の進捗）

- 高さ 4px、track `--surface-2`、fill `--text-1`（ライト）／ `--text-1`（ダーク）
- パーセンテージ表示は tabular-nums 必須。`80% — レイヤー 4/5 処理中` の併記必須
- 装飾的なアニメーション（流れる縞・グラデ）禁止
- **設計意図**：進捗バーは「処理が確実に進んでいる」唯一の濃要素として `--text-1` を許容する（職人道具の確実性）。これ以外の長尺な濃色塗り潰しは UI に置かない

### 4.10 LayerListItem

- 高さ 36px、左に 16×16 サムネ（ユーザー画像クロップ）
- 右に表示/非表示トグル（`eye` / `eye-off` Lucide アイコン、線 1.5px）
- 選択時：`bg: --surface-2`、左 2px の `--text-1` ストライプ

### 4.11 Dropzone（メインの画像投入領域）

- 1px dashed `--border`、角丸 `--radius-lg`、内側 padding 32px
- Hover / Drag-over：`border-color: --border-strong; bg: --surface-1`
- メッセージは和文 14px / 欧文 13px、tertiary text。**装飾アイコンを使わない**（テキストと薄い枠のみで完結）

### 4.12 Iconography

- ライブラリ：**Lucide**（線 1.5px、24px ベース・16px / 20px サイズ展開）
- 色は文字と同じトークン（`--text-1` / `--text-2`）
- **絵文字・カラフルアイコン・3D アイコン・装飾的グリフ禁止**

---

## 5. Layout Principles

### 5.1 Grid

- ベース単位：**4px**。すべての margin / padding / gap は 4 / 8 / 12 / 16 / 24 / 32 / 48px から選ぶ
- 中間値（10px / 18px / 22px 等）は禁止

### 5.2 メイン画面（3 カラム）

```
┌────────────────────────────────────────────────────────────┐
│ TopBar 40px  │ menu / language / settings                  │
├──────────┬──────────────────────────┬──────────────────────┤
│ LeftRail │   ImagePreview (flex)    │  LayerPanel 280px    │
│ 56px     │                          │                      │
│ (tools)  │                          │                      │
├──────────┴──────────────────────────┴──────────────────────┤
│ ControlBar 56px  │ model / pattern / [Decompose] button    │
├────────────────────────────────────────────────────────────┤
│ StatusBar 28px   │ progress / log tail                     │
└────────────────────────────────────────────────────────────┘
```

### 5.3 ブレークポイント（デスクトップ専用）

| BP | 幅 | 挙動 |
|---|---|---|
| Min | 800×600 | LeftRail を非表示、LayerPanel を 240px に縮小 |
| Default | 1280×800 | 上記 3 カラム標準 |
| Wide | 1920+ | ImagePreview の余白を増やす（上限 1600px キャンバス） |

### 5.4 余白規則

- パネル間 gap：`1px`（境界線で表現）
- パネル内コンテンツ padding：`12px 16px`
- セクション間：`24px`

---

## 6. Depth & Elevation

| Token | 用途 | 値（Light） | 値（Dark） |
|---|---|---|---|
| `--elevation-0` | パネル・カード（既定） | none（border のみ） | none（border のみ） |
| `--elevation-1` | Hover / Toast | `0 1px 2px rgba(28,26,22,0.06)` | `0 1px 2px rgba(0,0,0,0.4)` |
| `--elevation-2` | Dropdown / Popover | `0 4px 12px rgba(28,26,22,0.08)` | `0 4px 12px rgba(0,0,0,0.5)` |
| `--elevation-3` | Modal | `0 12px 32px rgba(28,26,22,0.12)` | `0 12px 32px rgba(0,0,0,0.6)` |

**原則**：階層はまずヘアラインで表現する。影は最小限・直角気味（blur ≤ 32px、spread 0）。**ガウシアンの強い「ふわっとした」影は禁止**。

---

## 7. Do's and Don'ts

### 7.1 Do（必ずやる）

- ユーザー画像を最も発色させ、UI は背景に退かせる
- ヘアラインで階層を表現する（影に頼らない）
- 数値（進捗・寸法・%）は `tabular-nums` で揃える
- 和文の line-height は 1.6 以上を維持
- 色の意味を一意に保つ（Success/Warning/Error は状態通知のみ）
- フォーカスリングは常に可視（`--focus` / 2px outline / 2px offset）
- アイコンは Lucide 線画モノクロのみ

### 7.2 Don't（絶対にやらない）

以下は **AI 生成 UI に頻発する典型** であり、本プロダクトでは禁止する：

- ❌ **紫グラデ＋白カード**（Stripe / 一般的な SaaS マーケサイト風）
- ❌ **ストックフォト**（笑顔の人物写真・抽象的な「テクノロジー」画像）
- ❌ **装飾アイコン**（絵文字・3D グリフ・カラフルなフラットアイコン）
- ❌ グロウ / ネオン / 強い blur / glassmorphism
- ❌ 過度なグラデーション（特にブランドカラー間のグラデ）
- ❌ 純黒 `#000000` / 純白 `#FFFFFF`（紙と現像トレイの質感が消える）
- ❌ 角丸 8px 以上（道具感が失われる）
- ❌ 装飾アニメーション（streaming-stripe progress / pulse / floating particles）
- ❌ Success/Warning/Error カラーを装飾用途に流用
- ❌ 「AI で〜」を煽る紫青のヒーローセクション風モーダル
- ❌ Noto Sans JP の 900、`palt` を本文に適用
- ❌ Material Design 風のフローティング FAB
- ❌ `font-family` の単段指定（必ず 3 段以上のフォールバック）

---

## 8. Responsive Behavior

本アプリは**デスクトップ専用**（Tauri ネイティブ）。Web のレスポンシブとは扱いが異なる。

| 観点 | 方針 |
|---|---|
| 最小ウィンドウ | 800×600（`SPECIFICATION.md §14.2` と整合） |
| リサイズ | 連続的にフレキシブル。レイアウトは §5.3 のブレークポイントで切替 |
| タッチ対応 | しない（マウス + キーボード前提） |
| DPI | 1.0× / 1.25× / 1.5× / 2.0× を考慮。ヘアラインは `0.5px` ではなく `1px` 固定 |
| OS テーマ | `prefers-color-scheme` 追従（MVP は system 固定。手動切替は Post-MVP） |
| Reduced Motion | `prefers-reduced-motion: reduce` 時、トランジション 0ms にフォールバック |
| 高コントラスト | OS 設定の High Contrast 時、`--border` を `--border-strong` に格上げ |

---

## 9. Agent Prompt Guide

### 9.1 新規コンポーネントを作る AI への定型プロンプト

```
DESIGN.md に従って <ComponentName> を実装してください。
- 色は §2 のトークン（CSS 変数）のみ使用、生 hex 禁止
- 角丸は 2 / 4 / 6px のいずれか
- 影は §6 の elevation トークンのみ
- 和文の line-height は 1.6 以上
- アイコンは Lucide 線画 1.5px
- §7.2 の Don't に該当するスタイルを使わない
- ライト・ダーク両テーマで動作確認（システムテーマ追従）
```

### 9.2 Tailwind v4 トークンマッピング（参考）

```css
/* src/frontend/src/styles/tokens.css */
@theme {
  --color-canvas: var(--canvas);
  --color-surface-1: var(--surface-1);
  --color-surface-2: var(--surface-2);
  --color-border: var(--border);
  --color-border-strong: var(--border-strong);
  --color-text-1: var(--text-1);
  --color-text-2: var(--text-2);
  --color-text-3: var(--text-3);
  --color-accent: var(--accent);
  --color-accent-fg: var(--accent-fg);
  --color-success: var(--success);
  --color-warning: var(--warning);
  --color-error: var(--error);
  --color-focus: var(--focus);
  --radius-sm: 2px;
  --radius-md: 4px;
  --radius-lg: 6px;
  --font-sans: "Inter", "Noto Sans JP", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", monospace;
  --shadow-elevation-1: 0 1px 2px rgba(28, 26, 22, 0.06);
  --shadow-elevation-2: 0 4px 12px rgba(28, 26, 22, 0.08);
  --shadow-elevation-3: 0 12px 32px rgba(28, 26, 22, 0.12);
}

:root[data-theme="dark"] {
  --shadow-elevation-1: 0 1px 2px rgba(0, 0, 0, 0.4);
  --shadow-elevation-2: 0 4px 12px rgba(0, 0, 0, 0.5);
  --shadow-elevation-3: 0 12px 32px rgba(0, 0, 0, 0.6);
}
```

影は必ず `shadow-elevation-{1,2,3}` トークン経由で参照し、生 `box-shadow` の直書きを禁止する。

### 9.3 レビュアー（人間 / `@code-reviewer`）向けチェックリスト

- [ ] 生 hex / 生 px が CSS に直接書かれていない（トークン参照のみ）
- [ ] §7.2 の Don't 項目に該当するスタイルがない
- [ ] 和文 `line-height >= 1.6`、`#000000` / `#FFFFFF` 不使用
- [ ] `font-family` は 3 段以上のフォールバック
- [ ] フォーカスリングが全インタラクティブ要素に存在
- [ ] ライト・ダーク両テーマで WCAG AA 達成
- [ ] アイコンが Lucide 線画モノクロ
- [ ] 数値表示に `tabular-nums` 適用
- [ ] `prefers-reduced-motion` 対応

---

## 10. References & Lineage

### 10.1 参照すべき製品（系譜）

- **Adobe Photoshop / Lightroom / Affinity Photo**：暗いキャンバス・道具感・ユーザー画像最優先
- **Linear / Warp / Raycast**：プロツール密度・ヘアライン階層・キーボード前提
- **MUJI / Apple Japan**：和の落ち着き・無漂白の質感・装飾を削ぐ思想
- **Notion**（適度に参照）：可読性のための余白設計

### 10.2 参照しない（拒絶）製品

- Stripe・Linear のマーケトップ：ブランドグラデーション
- Cohere・OpenAI のマーケ：カラフルな AI 装飾
- Lovable・Bolt：プレイフルな AI コーディング系
- 一般的な SaaS ヒーローセクション：白背景＋紫グラデ＋ストックフォト

### 10.3 整合するプロジェクト内ドキュメント

- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) §14.1 メイン画面・§14.2 UI 方針・§10 セットアップ
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) Phase 2 フロントエンド MVP
- [docs/reference_doc/販売戦略.md](docs/reference_doc/販売戦略.md) ターゲット像・$9.9 全部入りメッセージ
- [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md) 設定値・ファイル配置
- [WHITEBOARD.md](WHITEBOARD.md) Phase 進捗

---

> **変更ポリシー**：本ドキュメントを変更する PR は必ず `@code-reviewer` のレビューを受け、§2 / §3 / §6 のトークン変更時は `src/frontend/src/styles/tokens.css` の同期更新を必須とする。
