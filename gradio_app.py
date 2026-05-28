try:
    from transformers.models.bert.modeling_bert import BertModel

    if not hasattr(BertModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            # 新しい transformers で削除された旧 helper を GroundingDINO 互換のために補う。
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask.to(dtype=self.dtype)

        BertModel.get_head_mask = get_head_mask
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------------------
# Modified from Grounded-SAM (https://github.com/IDEA-Research/Grounded-Segment-Anything)
# ------------------------------------------------------------------------
import os
import random
import cv2
from scipy import ndimage

import gradio as gr
import argparse

import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import networks
import utils
from pipelines.components.model_components import default_device, require_gpu_for_heavy_inference

# Grounding DINO
import sys
sys.path.insert(0, './GroundingDINO')
from groundingdino.util.inference import Model

# SAM
sys.path.insert(0, './segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide

# SD
from diffusers import StableDiffusionPipeline

transform = ResizeLongestSide(1024)
# Green Screen
PALETTE_back = (51, 255, 146)

GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"
mam_checkpoint="checkpoints/mam_vitb.pth"
output_dir="outputs"
device = default_device()
require_gpu_for_heavy_inference("gradio_app.py", device)
background_list = os.listdir('assets/backgrounds')

# initialize MAM
mam_model = networks.get_generator_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')
mam_model.to(str(device))
checkpoint = torch.load(mam_checkpoint, map_location=device, weights_only=True)
mam_model.m2m.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
mam_model = mam_model.eval()

# initialize GroundingDINO
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=torch.device(device))

# initialize StableDiffusionPipeline
sd_dtype = torch.float16 if device.startswith("cuda") else torch.float32
generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=sd_dtype)
generator.to(str(device))

def run_grounded_sam(input_image, text_prompt, task_type, background_prompt, background_type, box_threshold, text_threshold, iou_threshold, scribble_mode, guidance_mode):

    global grounding_dino_model, generator

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load image
    # Gradio 5 ImageEditor: background=元画像, layers=[スクリブルレイヤー], composite=合成画像
    if isinstance(input_image, dict):
        image_ori = input_image.get('background', input_image.get('composite', input_image.get('image')))
    else:
        image_ori = input_image
    if image_ori is None:
        raise ValueError("入力画像が取得できませんでした。")
    # RGBA → RGB（PNGアップロード時や gr.ImageEditor が 4ch で返す場合に対応）
    if isinstance(image_ori, np.ndarray) and image_ori.ndim == 3 and image_ori.shape[2] == 4:
        image_ori = image_ori[:, :, :3]
    if isinstance(input_image, dict):
        if 'layers' in input_image and len(input_image['layers']) > 0:
            mask_layer = input_image['layers'][0]
            scribble = mask_layer
        else:
            scribble = np.zeros((*image_ori.shape[:2], 4), dtype=np.uint8)
    else:
        scribble = np.zeros((*image_ori.shape[:2], 4), dtype=np.uint8)
    original_size = image_ori.shape[:2]

    if task_type == 'text':
        if not text_prompt:
            raise gr.Error("テキストプロンプトを入力してください。")
        with torch.no_grad():
            detections, phrases = grounding_dino_model.predict_with_caption(
                image=cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR),
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

        if len(detections.xyxy) > 1:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                iou_threshold,
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]

        if len(detections.xyxy) == 0:
            raise gr.Error("テキストプロンプトに一致する領域が検出されませんでした。")
        bbox = detections.xyxy[np.argmax(detections.confidence)]
        bbox = transform.apply_boxes(bbox, original_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(str(device))

    image = transform.apply_image(image_ori)
    image = torch.as_tensor(image).to(str(device))
    image = image.permute(2, 0, 1).contiguous()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).to(str(device))
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).to(str(device))

    image = (image - pixel_mean) / pixel_std

    h, w = image.shape[-2:]
    pad_size = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh))

    if task_type == 'scribble_point':
        scribble = scribble.transpose(2, 1, 0)[0]

        labeled_array, num_features = ndimage.label(scribble >= 255)

        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
        if len(centers) == 0:
            raise gr.Error("スクリブルが検出されませんでした。画像上に描画してください。")
        centers = np.array(centers)
        ### (x,y)
        centers = transform.apply_coords(centers, original_size)
        point_coords = torch.from_numpy(centers).to(str(device))
        point_coords = point_coords.unsqueeze(0).to(str(device))
        point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(str(device))
        if scribble_mode == 'split':
            point_coords = point_coords.permute(1, 0, 2)
            point_labels = point_labels.permute(1, 0)
            
        sample = {'image': image.unsqueeze(0), 'point': point_coords, 'label': point_labels, 'ori_shape': original_size, 'pad_shape': pad_size}
    elif task_type == 'scribble_box':
        scribble = scribble.transpose(2, 1, 0)[0]
        labeled_array, num_features = ndimage.label(scribble >= 255)
        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
        if len(centers) == 0:
            raise gr.Error("スクリブルが検出されませんでした。画像上に描画してください。")
        centers = np.array(centers)
        ### (x1, y1, x2, y2)
        x_min = centers[:, 0].min()
        x_max = centers[:, 0].max()
        y_min = centers[:, 1].min()
        y_max = centers[:, 1].max()
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox = transform.apply_boxes(bbox, original_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(str(device))

        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    elif task_type == 'text':
        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    else:
        raise gr.Error(f"無効な task_type: {task_type}")

    with torch.no_grad():
        feas, pred, post_mask = mam_model.forward_inference(sample)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)
        
        if guidance_mode == 'mask':
            weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
            post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
            alpha_pred = post_mask.clone().detach()
        else:
            weight_os8 = utils.get_unknown_box_from_mask(post_mask)
            alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
            alpha_pred = alpha_pred_os8.clone().detach()


        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
       
        alpha_pred = alpha_pred[0][0].cpu().numpy()

    #### draw
    ### alpha matte
    alpha_rgb = cv2.cvtColor(np.uint8(alpha_pred*255), cv2.COLOR_GRAY2RGB)
    ### com img with background
    if background_type == 'real_world_sample':
        background_img_file = os.path.join('assets/backgrounds', random.choice(background_list))
        background_img = cv2.imread(background_img_file)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
        com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
        com_img = np.uint8(com_img)
    else:
        if not background_prompt:
            raise gr.Error("背景プロンプトを入力してください。")
        else:
            background_img = generator(background_prompt).images[0]
            background_img = np.array(background_img)
            background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
            com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
            com_img = np.uint8(com_img)
    ### com img with green screen
    green_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
    green_img = np.uint8(green_img)
    return [(com_img, 'composite with background'), (green_img, 'green screen'), (alpha_rgb, 'alpha matte')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument('--port', type=int, default=7589, help='port to run the server')
    parser.add_argument('--no-gradio-queue', action="store_true", help='path to the SAM checkpoint')
    args = parser.parse_args()

    print(args)

    with gr.Blocks() as block:
        gr.Markdown(
        """
        # Matting Anything Demo
        Welcome to the Matting Anything demo and upload your image to get started <br/> You may select different prompt types to get the alpha matte of target instance, and select different backgrounds for image composition.
        ## Usage
        You may check the <a href='https://www.youtube.com/watch?v=XY2Q0HATGOk'>video</a> to see how to play with the demo, or check the details below.
        <details>
        You may upload an image to start, we support 3 prompt types to get the alpha matte of the target instance：

        **scribble_point**: Click an point on the target instance.

        **scribble_box**: Click on two points, the top-left point and the bottom-right point to represent a bounding box of the target instance.

        **text**: Send text prompt to identify the target instance in the `Text prompt` box.

        We also support 2 background types to support image composition with the alpha matte output:

        **real_world_sample**: Randomly select a real-world image from `assets/backgrounds` for composition.

        **generated_by_text**: Send background text prompt to create a background image with stable diffusion model in the `Background prompt` box.
        </details>
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.ImageEditor(type="numpy", value="assets/demo.jpg", label="Upload Image")
                task_type = gr.Dropdown(["scribble_point", "scribble_box", "text"], value="text", label="Prompt type")
                text_prompt = gr.Textbox(label="Text prompt", placeholder="the girl in the middle")
                background_type = gr.Dropdown(["generated_by_text", "real_world_sample"], value="generated_by_text", label="Background type")
                background_prompt = gr.Textbox(label="Background prompt", placeholder="downtown area in New York")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                    )
                    scribble_mode = gr.Dropdown(
                        ["merge", "split"], value="split", label="scribble_mode"
                    )
                    guidance_mode = gr.Dropdown(
                        ["mask", "alpha"], value="alpha", label="guidance_mode", info="mask guidance is for complex scenes with multiple instances, alpha guidance is for simple scene with single instance"
                    )

            with gr.Column():
                gallery = gr.Gallery(
                    label="Generated images", show_label=True, elem_id="gallery"
                )

        run_button.click(fn=run_grounded_sam, inputs=[
                        input_image, text_prompt, task_type, background_prompt, background_type, box_threshold, text_threshold, iou_threshold, scribble_mode, guidance_mode], outputs=gallery)

    if not args.no_gradio_queue:
        block.queue()
    block.launch(server_name='0.0.0.0', server_port=args.port, debug=args.debug, share=args.share)
