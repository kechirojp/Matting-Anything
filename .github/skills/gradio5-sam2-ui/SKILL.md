---
name: gradio5-sam2-ui
description: Use this when changing Gradio 5 based SAM2 UI, prompt canvas, event wiring, dataframe handlers, or when validating UI behavior with Playwright.
---

# Skill: Gradio 5 + SAM2 UI

## Scope

この skill は Gradio 5 / SAM2 UI 層の改修時に使う。対象は UI 構成、イベント配線、実行時検証、動画 UI のフレーム選択導線。

## Core Rules

- `gr.Image(tool="sketch")` は使わず `gr.ImageEditor` を使う（ERR002）。
- `block = block.queue()` は禁止。`with gr.Blocks() as block:` の後に `block.queue()` を呼ぶ（ERR001）。
- `ImageEditor` 戻り値は `background/layers/composite` を使う（ERR003）。
- RGBA はモデル入力前に RGB へ変換する（ERR008）。
- SAM2 bbox / point の手入力 UI は禁止。画像クリック由来に限定する（ERR017）。
- Point 正負は `gr.Radio(["positive", "negative"])` で明示する（ERR019）。
- Prompt Canvas は `sources=[]` を維持し、アップロード欄と分離する（ERR021, ERR026, ERR031）。
- UI / 配線の fixed 記録前に Gradio 実起動 + Playwright 実行時確認を必須とする（ERR035）。
- Gradio 5/Svelte では DOM 直接 `value=` + native event dispatch の JS ブリッジを使わない（ERR035）。
- `gr.Dataframe` 値は pandas DataFrame を想定し、真偽評価 (`rows or []`, `if rows`) をしない（ERR036）。
- スライダー上限と実処理レンジが乖離する場合は、重い処理前に fail-fast で `gr.Error` を返す（ERR037）。

## Video UI Specific

- 「表示中フレーム再取得」等の冗長コントロールは避け、単一操作元へ集約する（ERR033, ERR035）。
- tracker/background dropdown は埋没させず、主要導線に配置する（ERR032）。
- `prompt_frame_idx` は実サンプリング枚数と整合したガードを持つ（ERR037）。

## Error Coverage

ERR001, ERR002, ERR003, ERR008, ERR011, ERR015, ERR016, ERR017, ERR019, ERR021, ERR026, ERR031, ERR032, ERR033, ERR035, ERR036, ERR037

## Validation Checklist

- Gradio アプリは `--help` で起動確認できる。
- Playwright で想定操作に反応する（例: slider drag -> prompt canvas update）。
- `WHITEBOARD.md` / `ERROR_LOG.md` の fixed 記録前に実行時確認ログを残す。
