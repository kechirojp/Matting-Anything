# Pipeline Assembly

## 基本手順

1. `Pipeline()` を作る。
2. `add_component(name, component)` で Component を登録する。
3. `connect("source.output", "target.input")` で型付き socket を結ぶ。
4. Gradio からは `pipeline.run({"component": {"input": value}})` で実行する。

## Matting の典型構成

MAM:

```text
ImageNormalizer -> MAMAlphaPredictor -> BackgroundGenerator
                                      -> OutputSaver
```

SAM2 + transparent-background:

```text
ImageNormalizer -> TransparentBGExtractor -> SAM2GuardFilter -> ColorDecontaminator -> OutputSaver
```

## 分岐の扱い

- Gradio UI の選択値は Pipeline 入力として渡す。
- モデル差し替えは Component インスタンスの差し替えで行う。
- DAG トポロジ変更は最後の手段にする。