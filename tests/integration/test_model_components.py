import pytest


@pytest.mark.integration
def test_mam_pipeline_requires_checkpoints() -> None:
    from pipelines.mam_pipeline import build_mam_pipeline

    pipeline = build_mam_pipeline()
    assert pipeline is not None


@pytest.mark.integration
def test_sam2_tb_pipeline_requires_sam2_and_transparent_background() -> None:
    from pipelines.sam2_tb_pipeline import build_sam2_tb_pipeline

    pipeline = build_sam2_tb_pipeline()
    assert pipeline is not None