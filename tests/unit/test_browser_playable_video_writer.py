"""alpha/preview/overlay 動画がブラウザ(gr.Video)再生互換な VP9(webm) で書き出されることを検証する。

ERR065: cv2 'mp4v'(MPEG-4 Part 2) は Chrome 等の <video>/gr.Video で再生できず、出力が
「表示されない」ように見える。RGBA(webm/VP9) と同様に alpha/preview/tracking overlay も
VP9(yuv420p)/webm で書き出し、全 Chromium 系（Playwright を含む）で再生互換にする。
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pipelines.components.video_model_components import _ImageioWebmVideoWriter


def _probe_stream_info(path) -> str:
    """imageio-ffmpeg 同梱の ffmpeg でストリーム情報文字列を得る。"""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    return (proc.stderr or "") + (proc.stdout or "")


def test_webm_writer_outputs_browser_playable_codec(tmp_path):
    path = tmp_path / "preview.webm"
    frame = np.full((18, 32, 3), 120, np.uint8)
    writer = _ImageioWebmVideoWriter(path, frame, 30.0)
    for _ in range(4):
        writer.write(frame)
    writer.close()
    assert path.exists() and path.stat().st_size > 0
    info = _probe_stream_info(path).lower()
    assert "vp9" in info, f"VP9 で書き出されていません: {info}"
    assert "mpeg-4 part 2" not in info and "mpeg4" not in info


def test_webm_writer_accepts_grayscale_alpha_frames(tmp_path):
    path = tmp_path / "alpha.webm"
    gray = np.full((18, 32), 200, np.uint8)
    writer = _ImageioWebmVideoWriter(path, gray, 30.0)
    for _ in range(4):
        writer.write(gray)
    writer.close()
    assert path.exists() and path.stat().st_size > 0
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="FFMPEG")
    try:
        first = reader.get_data(0)
    finally:
        reader.close()
    # grayscale は RGB(3ch) に複製して書き出す（yuv420p 互換）。
    assert first.ndim == 3 and first.shape[2] == 3
