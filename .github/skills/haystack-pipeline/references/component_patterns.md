# Component Patterns

## 基本形

```python
from haystack import component


@component
class ImageNormalizer:
    @component.output_types(image=np.ndarray)
    def run(self, image: dict | np.ndarray | Image.Image) -> dict[str, np.ndarray]:
        return {"image": normalized}
```

## 重いモデル

- `__init__`: checkpoint path、device、設定値だけ保持する。
- `warm_up()`: モデル生成、checkpoint load、`eval()` を行う。
- `run()`: `warm_up()` 済みでなければ呼び出し、`torch.no_grad()` / `torch.inference_mode()` で推論する。

## 状態管理

- モデルインスタンスは Component の private 属性に置く。
- 永続状態は checkpoint や `outputs/` の artifact に出す。
- Pipeline のシリアライズを壊すため、巨大 tensor や optimizer state を設定辞書に含めない。

## 例外

- Component 内では根本原因が分かる `ValueError` / `RuntimeError` を raise する。
- Gradio 境界で `gr.Error` に変換し、ユーザーに日本語メッセージを出す。