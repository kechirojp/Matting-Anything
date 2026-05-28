# Gradio Integration

## 方針

Gradio は UI、Haystack は推論 DAG。Gradio コールバックは次だけを担当する。

1. UI 入力を Pipeline 入力 dict に詰める。
2. `pipeline.run(...)` を呼ぶ。
3. 出力を Gradio 表示形式へ整える。
4. 例外を `gr.Error` に変換する。

## ImageEditor

Gradio 5 の `ImageEditor` は `background` / `layers` / `composite` を返す。Component 境界では必ず RGB `np.ndarray` に正規化する。

```python
try:
    result = pipeline.run({...})
except Exception as exc:
    raise gr.Error(str(exc)) from exc
```

## queue

`with gr.Blocks() as demo:` の後に `demo.queue()` を呼び、`demo.launch(...)` する。