from pathlib import Path

import numpy as np


def test_haystack_notebook_sources_are_jupytext_percent_files() -> None:
    for filename in ("Matting_Anything_Haystack.py", "Sam2_Transparent_Background_Haystack.py"):
        source = Path(filename).read_text(encoding="utf-8")
        assert "format_name: percent" in source
        assert "jupytext" in source


def test_haystack_notebook_sources_launch_expected_apps() -> None:
    mam_source = Path("Matting_Anything_Haystack.py").read_text(encoding="utf-8")
    sam2_source = Path("Sam2_Transparent_Background_Haystack.py").read_text(encoding="utf-8")
    movie_source = Path("Sam2_Transparent_Background_Haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "gradio_app_haystack.py" in mam_source
    assert "gradio_app_sam2_transparent_BG_haystack.py" in sam2_source
    assert "--share" in mam_source
    assert "--share" in sam2_source
    assert "--share" in movie_source


def test_colab_gradio_launch_cells_preserve_gradio_share_defaults() -> None:
    """Colab では Gradio の share 生成に任せ、public URL を開く案内を出す。"""
    notebook_sources = [
        Path("Matting_Anything_Haystack.py").read_text(encoding="utf-8"),
        Path("Sam2_Transparent_Background_Haystack.py").read_text(encoding="utf-8"),
        Path("Sam2_Transparent_Background_Haystack_for_Movie.py").read_text(encoding="utf-8"),
    ]

    for source in notebook_sources:
        assert "def is_colab_runtime()" in source
        assert 'find_spec("google.colab")' in source
        assert "except (ModuleNotFoundError, ValueError):" in source
        assert source.count("def is_colab_runtime()") == 1
        assert 'SHARE_FLAG = "--share" if IS_COLAB else ""' in source
        assert "Running on public URL" in source
        assert "gradio.live" in source
        assert "not the local 127.0.0.1 URL" in source
        assert "ensure_gradio_share_binary_for_colab" not in source
        assert "checksum verification failed" not in source


def test_sam2_haystack_notebook_installs_groundingdino_runtime_dependencies() -> None:
    """Text Prompt 検出に必要な GroundingDINO 依存が Colab install cell に含まれることを担保する。"""
    sam2_source = Path("Sam2_Transparent_Background_Haystack.py").read_text(encoding="utf-8")

    for package_name in ("supervision", "addict", "yapf", "timm", "pycocotools"):
        assert package_name in sam2_source
    assert "transformers>=4.26.0" in sam2_source


def test_runtime_requirements_match_groundingdino_api_assumptions() -> None:
    """GroundingDINO patches rely on modern transformers and PyTorch APIs."""
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "torch>=2.0.0" in requirements
    assert "transformers>=4.26.0" in requirements


def test_sam2_haystack_notebook_has_colab_device_diagnostics() -> None:
    """Colab 上で Notebook kernel と Gradio process の CUDA / checkpoint 状態を確認できる。"""
    sam2_source = Path("Sam2_Transparent_Background_Haystack.py").read_text(encoding="utf-8")

    assert "nvidia-smi" in sam2_source
    assert "torch.cuda.is_available()" in sam2_source
    assert "torch.version.cuda" in sam2_source
    assert "GROUNDING_DINO_CKPT_PATH" in sam2_source
    assert 'os.environ["GROUNDING_DINO_CKPT_PATH"]' in sam2_source
    assert "SAM2_CKPT_PATH" in sam2_source
    assert "MATTING_ANYTHING_ALLOW_CPU=1" in sam2_source
    assert "emergency CPU fallback" in sam2_source
    assert "CUDA GPU runtime is required before launching Gradio" in sam2_source
    assert "ランタイムのタイプを変更" in sam2_source
    assert "SAM2 package import failed before launching Gradio" in sam2_source
    assert "from sam2.build_sam import build_sam2" in sam2_source
    assert "from sam2.sam2_image_predictor import SAM2ImagePredictor" in sam2_source


def test_sam2_movie_notebook_has_colab_device_diagnostics() -> None:
    """Movie notebook should expose CUDA/checkpoint diagnostics before launching Gradio."""
    movie_source = Path("Sam2_Transparent_Background_Haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "nvidia-smi" in movie_source
    assert "torch.cuda.is_available()" in movie_source
    assert "torch.version.cuda" in movie_source
    assert "SAM2 checkpoint:" in movie_source
    assert "GroundingDINO checkpoint:" in movie_source
    assert 'os.environ["SAM2_CKPT_PATH"]' in movie_source
    assert 'os.environ["GROUNDING_DINO_CKPT_PATH"]' in movie_source
    assert "MATTING_ANYTHING_ALLOW_CPU=1" in movie_source
    assert "emergency CPU fallback" in movie_source
    assert "CUDA GPU runtime is required before launching Gradio" in movie_source
    assert "ランタイムのタイプを変更" in movie_source
    assert "SAM2 package import failed before launching Gradio" in movie_source
    assert "from sam2.build_sam import build_sam2_video_predictor" in movie_source


def test_sam2_haystack_app_patches_gradio_bool_schema_issue() -> None:
    from gradio_app_sam2_transparent_BG_haystack import _patched_json_schema_to_python_type

    assert _patched_json_schema_to_python_type(True) == "Any"
    assert _patched_json_schema_to_python_type(False) == "Any"
    nested_result = _patched_json_schema_to_python_type({"type": "object", "additionalProperties": True})
    assert isinstance(nested_result, str)
    assert "Any" in nested_result


def test_sam2_haystack_box_prompt_snaps_to_image_edges() -> None:
    from gradio_app_sam2_transparent_BG_haystack import normalize_box_from_points

    box = normalize_box_from_points((4, 5), (635, 474), image_shape=(480, 640, 3))

    assert box == [0, 0, 639, 479]


def test_sam2_haystack_app_has_no_manual_sam2_coordinate_inputs() -> None:
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert "Point X" not in app_source
    assert "Point Y" not in app_source
    assert "gr.Number" not in app_source


def test_sam2_haystack_select_prompt_builds_box_from_mouse_clicks() -> None:
    from gradio_app_sam2_transparent_BG_haystack import select_sam2_prompt

    class FakeSelectData:
        def __init__(self, index):
            self.index = index

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    state = {}

    _preview, state, _status = select_sam2_prompt(image, "box", True, state, FakeSelectData((2, 3)))
    assert state["box"] is None
    assert state["box_buffer"] == [(0, 0)]

    _preview, state, status = select_sam2_prompt(image, "box", True, state, FakeSelectData((119, 99)))
    assert state["box"] == [0, 0, 119, 99]
    assert state["box_buffer"] == []
    assert "Box" in status


def test_sam2_haystack_app_requests_intermediate_pipeline_outputs() -> None:
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert "include_outputs_from" in app_source
    assert "transparent_bg" in app_source
    assert "sam2_guard" in app_source
    assert "output_saver" in app_source


def test_sam2_haystack_extend_box_to_edge_modifies_each_side() -> None:
    """4 つのエッジボタンが bbox の対応する辺を画像端へ延長することを検証する。"""
    from gradio_app_sam2_transparent_BG_haystack import extend_box_to_edge

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    base_state = {"box": [30, 40, 80, 60]}

    _preview, state_left, _ = extend_box_to_edge(image, base_state, "left")
    assert state_left["box"] == [0, 40, 80, 60]

    _preview, state_right, _ = extend_box_to_edge(image, base_state, "right")
    assert state_right["box"] == [30, 40, 119, 60]

    _preview, state_top, _ = extend_box_to_edge(image, base_state, "top")
    assert state_top["box"] == [30, 0, 80, 60]

    _preview, state_bottom, _ = extend_box_to_edge(image, base_state, "bottom")
    assert state_bottom["box"] == [30, 40, 80, 99]


def test_sam2_haystack_extend_box_to_edge_requires_existing_box() -> None:
    """bbox 未確定時のエッジ延長要求は gr.Error を送出する。"""
    import gradio as gr

    from gradio_app_sam2_transparent_BG_haystack import extend_box_to_edge

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    try:
        extend_box_to_edge(image, {"box": None}, "left")
    except gr.Error:
        return
    raise AssertionError("bbox がない場合は gr.Error を送出すべき")


def test_sam2_haystack_app_uses_positive_negative_radio() -> None:
    """positive / negative の選択 UI が Radio として明示されていることを担保する。"""
    import re

    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    # gr.Radio(...) の choices に positive / negative の両方が並ぶ呼び出しがあること
    assert re.search(
        r"gr\.Radio\([^)]*\"positive\"[^)]*\"negative\"",
        app_source,
        flags=re.DOTALL,
    ), "positive / negative の Radio が見つかりません"


def test_sam2_haystack_app_uses_image_for_prompt_input() -> None:
    """SAM2 prompt 用入力は gr.Image(interactive=True) を使う。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert "interactive=True" in app_source
    assert "gr.ImageEditor(" not in app_source


def test_sam2_haystack_app_has_edge_extend_buttons() -> None:
    """4 つのエッジ延長ボタンが UI に存在することを担保する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    for side in ("left", "right", "top", "bottom"):
        assert f'"{side}"' in app_source, f"edge button for {side} missing"


def test_sam2_haystack_app_has_text_prompt_candidate_and_union_controls() -> None:
    """複合対象向けの text prompt / candidate / union UI が存在することを担保する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert 'label="Text Prompt"' in app_source
    assert "Detect Text Boxes" in app_source
    assert "Candidate Mask Indices" in app_source
    assert "Union Mask Preview" in app_source
    assert "Run transparent-background" in app_source
    assert "add_candidates_to_union" in app_source
    assert "remove_candidates_from_union" in app_source
    assert "clear_union_mask" in app_source


def test_sam2_haystack_app_has_japanese_required_optional_flow_help() -> None:
    """上から全部入力必須ではないことを日本語ガイドで説明する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert "すべてを上から順に入力する必要はありません" in app_source
    assert "必須" in app_source
    assert "任意" in app_source
    assert "最短フロー" in app_source


def test_sam2_haystack_app_tracks_candidate_mask_set_contract() -> None:
    """UI state が単一 best mask ではなく MaskSet / union mask を扱うことを担保する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert '"mask_set"' in app_source
    assert '"union_mask"' in app_source
    assert "build_mask_set" in app_source
    assert "union_masks" in app_source


def test_sam2_haystack_app_has_separate_prompt_canvas() -> None:
    """アップロード用 Image と SAM2 prompt 編集用 Image が分離されていることを担保する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert 'label="Input Image"' in app_source
    assert 'label="SAM2 Prompt Canvas"' in app_source
    assert "prompt_canvas.select(" in app_source
    assert "input_image.select(" not in app_source


def test_sam2_haystack_prompt_canvas_disables_upload_sources() -> None:
    """SAM2 Prompt Canvas はアップロード先ではなく、クリック可能な prompt 編集面にする。"""
    import re

    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    prompt_canvas_match = re.search(r"prompt_canvas = gr\.Image\((.*?)\n\s+\)", app_source, flags=re.DOTALL)
    assert prompt_canvas_match is not None
    prompt_canvas_source = prompt_canvas_match.group(1)

    assert "value=create_prompt_canvas_placeholder()" in prompt_canvas_source
    assert 'label="SAM2 Prompt Canvas"' in prompt_canvas_source
    assert "sources=[]" in prompt_canvas_source
    assert "interactive=True" in prompt_canvas_source
    assert "interactive=False" not in prompt_canvas_source
    assert "show_download_button=False" in prompt_canvas_source
    assert "show_fullscreen_button=False" in prompt_canvas_source


def test_sam2_haystack_prompt_canvas_sync_resets_prompt_state() -> None:
    """入力画像を編集キャンバスへ同期すると prompt と SAM2 mask が初期化される。"""
    from gradio_app_sam2_transparent_BG_haystack import SAM2_STATE, sync_prompt_canvas

    image = np.zeros((24, 32, 3), dtype=np.uint8)
    SAM2_STATE["mask"] = np.ones((24, 32), dtype=bool)

    canvas, state, status = sync_prompt_canvas(image)

    assert canvas.shape == image.shape
    assert canvas is not image
    assert state["points"] == []
    assert state["box"] is None
    assert SAM2_STATE == {}
    assert "SAM2 Prompt Canvas" in status


def test_sam2_haystack_prompt_canvas_has_non_upload_placeholder() -> None:
    """空の SAM2 Prompt Canvas はアップロードUIではなく説明入り画像を表示する。"""
    from gradio_app_sam2_transparent_BG_haystack import create_prompt_canvas_placeholder, sync_prompt_canvas

    placeholder = create_prompt_canvas_placeholder()
    canvas, state, status = sync_prompt_canvas(None)

    assert placeholder.ndim == 3
    assert placeholder.shape == canvas.shape
    assert state["points"] == []
    assert status == "Input image is empty."


def test_sam2_haystack_image_display_size_updates_default_window() -> None:
    """予測画像はデフォルトをウィンドウサイズにし、オリジナルサイズへ切替できる。"""
    from gradio_app_sam2_transparent_BG_haystack import OUTPUT_WINDOW_HEIGHT, PROMPT_CANVAS_HEIGHT, update_image_display_size

    window_updates = update_image_display_size("window")
    original_updates = update_image_display_size("original")

    assert len(window_updates) == 6
    assert len(original_updates) == 6
    assert all(update["height"] in (PROMPT_CANVAS_HEIGHT, OUTPUT_WINDOW_HEIGHT) for update in window_updates)
    assert all(update["height"] is None for update in original_updates)


def test_sam2_haystack_ui_reduces_required_union_and_text_inputs() -> None:
    """静止画 UI は Text Prompt / Union を任意導線にし、best candidate で実行できる。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack.py").read_text(encoding="utf-8")

    assert 'with gr.Accordion("Optional: Text Prompt to Box", open=False)' in app_source
    assert 'with gr.Accordion("Optional: Mask Union for composite subjects", open=False)' in app_source
    assert 'value="Best Candidate Mask"' in app_source
    assert 'placeholder="person playing drums / person riding bicycle"' in app_source
    assert 'value="person . object ."' not in app_source


def test_sam2_haystack_ui_hides_point_label_for_box_mode() -> None:
    """box prompt では point の positive/negative 選択を必須に見せない。"""
    from gradio_app_sam2_transparent_BG_haystack import update_point_label_visibility

    assert update_point_label_visibility("box")["visible"] is False
    assert update_point_label_visibility("point")["visible"] is True


def test_sam2_movie_ui_auto_loads_first_frame_and_simplifies_settings() -> None:
    """動画 UI はアップロード後の第1フレーム自動取得と詳細設定折りたたみを提供する。"""
    app_source = Path("gradio_app_sam2_transparent_BG_haystack_for_Movie.py").read_text(encoding="utf-8")

    assert "extract_first_frame_outputs" in app_source
    assert "input_video.change(extract_first_frame_outputs" in app_source
    assert 'gr.Button("第1フレームを再取得")' in app_source
    assert 'with gr.Accordion("Advanced: 動画処理設定", open=False)' in app_source
    assert 'prompt_mode = gr.Radio(["point", "box"], value="box"' in app_source
    assert 'max_frames = gr.Slider(1, 2000, value=30' in app_source
    assert 'frame_step = gr.Slider(1, 10, value=1' in app_source
    assert "SAM2 Prompt Canvas" in app_source
    assert "対角 2 点" in app_source
    assert "Text Prompt（意味解釈）→ SAM2（マスク/トラッキング）→ transparent-background" in app_source
    assert "パラメーター" in app_source


def test_sam2_movie_ui_hides_codec_for_sequence_and_point_label_for_box_mode() -> None:
    """動画 UI は sequence 時の codec と box 時の point label を不要入力として隠す。"""
    from gradio_app_sam2_transparent_BG_haystack_for_Movie import update_codec_visibility, update_point_label_visibility

    sequence_update = update_codec_visibility("連番静止画 (sequence)")
    video_update = update_codec_visibility("動画 (video)")

    assert sequence_update["interactive"] is False
    assert sequence_update["visible"] is False
    assert video_update["interactive"] is True
    assert video_update["visible"] is True
    assert update_point_label_visibility("box")["visible"] is False
    assert update_point_label_visibility("point")["visible"] is True