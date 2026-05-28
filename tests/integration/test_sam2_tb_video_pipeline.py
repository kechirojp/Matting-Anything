from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_sam2_tb_video_pipeline_output_modes_require_gpu_checkpoint() -> None:
    """SAM2 video predictor + transparent-background の実機確認用骨格。"""
    checkpoint = Path("checkpoints/SAM2/sam2.1_hiera_large.pt")
    if not checkpoint.exists():
        pytest.skip("SAM2 checkpoint がないため integration test を skip")
    pytest.skip("GPU 実機で短尺動画を用意して output_mode=video/sequence/both を確認する")
