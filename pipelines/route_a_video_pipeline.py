"""SAM2 + BEN2（ルートA: ブラー誘導 → 再α化）動画用 Haystack Pipeline。

既存の ``sam2_tb_video_pipeline`` と同一の配線・I/O 契約を保ちつつ、背景除去 Component だけを
``BEN2RouteAVideoExtractor`` に差し替える。これにより VideoWriter / FrameSequenceWriter /
TrackingOverlayWriter をそのまま再利用できる（疎結合・差し替え容易性）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from haystack import Pipeline

from .components.ben2_components import BEN2RouteAVideoExtractor
from .components.video_model_components import (
    FrameSequenceWriter,
    SAM2VideoPropagator,
    TrackingOverlayWriter,
    VideoReader,
    VideoWriter,
)

BEN2_COMPONENT_NAME = "ben2_route_a_video"


def build_ben2_route_a_only_video_pipeline() -> Pipeline:
    """SAM2 追跡なしで BEN2（ルートA）のみ動画処理する軽量 Pipeline を構築する。

    各フレームを mask 無し（全画面誘導なし）で BEN2 に渡す。単一 salient 対象など追跡が
    不要な用途向けの確認経路。
    """
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("ben2_route_a_video", BEN2RouteAVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())

    pipeline.connect("video_reader.frames", "ben2_route_a_video.frames")
    pipeline.connect("video_reader.metadata", "ben2_route_a_video.metadata")
    pipeline.connect("ben2_route_a_video.matte", "video_writer.matte")
    pipeline.connect("ben2_route_a_video.matte", "frame_sequence_writer.matte")
    return pipeline


def build_sam2_ben2_route_a_pipeline(
    propagator: SAM2VideoPropagator | None = None,
    extractor: BEN2RouteAVideoExtractor | None = None,
) -> Pipeline:
    """ルートA動画αマットの end-to-end Pipeline を構築する。

    配線は ``build_sam2_tb_video_pipeline`` と同型で、背景除去段のみ BEN2（ルートA）に置換する::

        VideoReader -> SAM2VideoPropagator -> OwnershipResolver -> BEN2RouteAVideoExtractor
                    -> VideoWriter / FrameSequenceWriter / TrackingOverlayWriter

    Args:
        propagator: 事前構築済みの SAM2VideoPropagator。tracker モデル選択を反映するために
            registry から構築したインスタンスを注入できる。None の場合は既定 SAM2 を構築する。
        extractor: 事前構築済みの BEN2RouteAVideoExtractor。BEN2 repo/checkpoint や
            output_dir を反映したインスタンスを注入できる。None の場合は既定を構築する。
    """
    pipeline = Pipeline()
    pipeline.add_component("video_reader", VideoReader())
    pipeline.add_component("sam2_video_propagator", propagator or SAM2VideoPropagator())
    # per-object logits -> soft ownership maps へ変換する（per_object 経路で利用）。
    from .components.ownership_resolver import OwnershipResolver

    pipeline.add_component("ownership_resolver", OwnershipResolver())
    pipeline.add_component("ben2_route_a_video", extractor or BEN2RouteAVideoExtractor())
    pipeline.add_component("video_writer", VideoWriter())
    pipeline.add_component("frame_sequence_writer", FrameSequenceWriter())
    pipeline.add_component("tracking_overlay", TrackingOverlayWriter())

    pipeline.connect("video_reader.frames", "sam2_video_propagator.frames")
    pipeline.connect("video_reader.metadata", "sam2_video_propagator.metadata")
    pipeline.connect("video_reader.frames", "ben2_route_a_video.frames")
    pipeline.connect("video_reader.metadata", "ben2_route_a_video.metadata")
    pipeline.connect("sam2_video_propagator.masks", "ownership_resolver.masks")
    pipeline.connect("ownership_resolver.masks", "ben2_route_a_video.masks")
    pipeline.connect("ben2_route_a_video.matte", "video_writer.matte")
    pipeline.connect("ben2_route_a_video.matte", "frame_sequence_writer.matte")
    pipeline.connect("video_reader.frames", "tracking_overlay.frames")
    pipeline.connect("video_reader.metadata", "tracking_overlay.metadata")
    pipeline.connect("sam2_video_propagator.masks", "tracking_overlay.masks")
    return pipeline


def warm_up_ben2_in_pipelines(
    pipelines: Iterable[Pipeline],
    *,
    log: Callable[[str], None] = print,
) -> int:
    """各 Pipeline 内の BEN2 extractor を事前ロードし、成功件数を返す。

    BEN2 の重み（``from_pretrained`` 経由・約380MB）は既定で初回推論時に遅延 DL される。
    Colab/gradio.live ではこの DL がリクエスト処理中（SSE ストリーム中）に走ると、HF 未認証
    リクエストのレート制限で DL が長時間化し SSE が無通信→idle 切断され全出力「Error」になる
    （ERR057＝ERR055/ERR056 follow-up）。``demo.launch()`` の **前**（SSE 接続が存在しない段階）
    に本関数で重みを取得しておけば、DL がどれだけ遅くても起動セル内で完結し、リクエスト時は
    キャッシュ済み重みを即時ロードできる。

    GPU/ben2 が無いローカル環境などで warm_up が失敗しても起動は止めない（リクエスト時の
    keep-alive フォールバックがあるため）。例外は握り潰さず ``log`` で必ず通知し、次の Pipeline
    の事前ロードを続行する。

    Args:
        pipelines: BEN2 extractor を含む Haystack Pipeline 群。
        log: 進捗/失敗を通知する callable（既定 ``print``、テストで差し替え可能）。

    Returns:
        事前ロードに成功した Pipeline 数。
    """
    warmed = 0
    for index, pipeline in enumerate(pipelines):
        try:
            component = pipeline.get_component(BEN2_COMPONENT_NAME)
            component.extractor.warm_up()
        except Exception as exc:  # noqa: BLE001 - 起動は止めず必ず通知し次を続行する
            log(f"[prewarm] BEN2 事前ロードに失敗しました（#{index}・リクエスト時に再試行します）: {exc}")
            continue
        warmed += 1
        log(f"[prewarm] BEN2 事前ロードが完了しました（#{index}）。")
    return warmed
