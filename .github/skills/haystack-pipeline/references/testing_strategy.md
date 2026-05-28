# Testing Strategy

## Unit tests

対象:

- ImageNormalizer
- ScribbleParser
- BBoxFromMask
- MaskDilator
- AlphaCompositor
- Pipeline wiring with lightweight Components

実行:

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration" -v
```

## Integration tests

対象:

- MAMAlphaPredictor
- GroundingDINODetector
- SAM2Segmenter
- TransparentBGExtractor
- Gradio app smoke tests

すべて `@pytest.mark.integration` を付ける。GPU、checkpoint、外部モデルが必要なため CI 既定では実行しない。