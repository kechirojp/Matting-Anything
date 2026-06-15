---
name: sam2-tracking-dino
description: Use this when changing SAM2/SAMURAI tracking, GroundingDINO integration, CUDA build behavior, Colab preflight, video propagation, or tracker registry wiring.
---

# Skill: SAM2 Tracking + GroundingDINO

## Scope

この skill は SAM2 / SAMURAI / GroundingDINO / 動画追跡のモデル層と実行層を扱う。Hydra 検索パス、CUDA build、動画伝搬、progress、メモリ管理を含む。

## Core Rules

- GroundingDINO の CUDA build は `os.environ` 設定 + `--no-build-isolation` を使う（ERR004, ERR010）。
- `patch_transformers_bert_for_groundingdino()` を import 前に適用する（ERR024）。
- `bertwarper.py` の attention mask 呼び出しは新シグネチャに合わせる（ERR005）。
- `ms_deform_attn.py` は `CUDA_OPS_AVAILABLE` ガードを使う（ERR006）。
- `checkpoint.checkpoint()` には `use_reentrant=False` を明示する（ERR007）。
- Colab Gradio 公開前に `sam2` import と依存 preflight を行い、欠落時は fail fast（ERR023, ERR025）。
- 動画版 progress は stage と frame 進捗を示し、固定 5% 放置を避ける（ERR029）。
- transparent-background 出力は frame 全保持を避け、逐次保存する（ERR030）。
- tracker 選択は pipeline 構築へ確実に伝搬する（ERR034）。
- SAMURAI config は `Path.as_uri()` で Hydra 検索パスへ追加し、`samurai/` 自体は変更しない（ERR038）。
- share link 依存欠落（frpc 等）は preflight で診断可能にする（ERR027）。

## Tracker / Registry

- registry id -> propagator の構築を明示し、既定 tracker 固定化を起こさない。
- `INFERENCE_TRACKER_VARIANT` による表示切替 (`sam2_facebook`, `sam2_samurai`) を壊さない。
- metadata に `tracker_config`, `tracker_checkpoint`, `samurai_mode` を残す。

## Error Coverage

ERR004, ERR005, ERR006, ERR007, ERR009, ERR010, ERR023, ERR024, ERR025, ERR027, ERR028, ERR029, ERR030, ERR034, ERR038

## Related Skills

- Haystack I/O 契約と pipeline 結線は `.github/skills/haystack-pipeline/SKILL.md` を併読する。
- UI/配線は `.github/skills/gradio5-sam2-ui/SKILL.md` を併読する。
