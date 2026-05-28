# Matting Components

| Component | 入力 | 出力 | 内部実装 | 差し替え候補 |
|-----------|------|------|----------|--------------|
| ImageNormalizer | dict / ndarray / PIL | RGB ndarray | Gradio 5 ImageEditor 正規化 | - |
| ScribbleParser | ImageEditor dict | points / box / mask | scipy.ndimage | Gradio event 入力 |
| BBoxFromMask | mask | bbox | numpy | - |
| MaskDilator | mask | mask | OpenCV | skimage morphology |
| AlphaCompositor | image / alpha / bg | composite / green | numpy | - |
| GroundingDINODetector | image / text | bbox | GroundingDINO | OWL-ViT |
| MAMAlphaPredictor | image / prompt | alpha | MAM + SAM v1 | 別 matting model |
| SAM2Segmenter | image / points / box | masks / scores | SAM2 | SAM v1 |
| TransparentBGExtractor | image / mask | rgba / alpha / preview | transparent-background | MAM |
| SAM2GuardFilter | alpha / mask | alpha | numpy + cv2 | - |
| ColorDecontaminator | image / alpha | rgb | pymatting | disable |
| BackgroundGenerator | image / alpha / prompt | composite | SD / asset | 任意生成器 |
| OutputSaver | rgba / alpha / preview | paths | PIL + outputs/ | artifact store |