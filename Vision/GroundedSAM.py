# code from groundedSAM and extra guidelines to install can be found in https://github.com/IDEA-Research/Grounded-Segment-Anything
# code has been modified to our purpose

import cv2
import sys
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyrealsense2 as rs

# GroundingDINO imports - add to sys.path if needed
sys.path.append("C:/Users/Gebruiker/GroundingDINO")

def get_3d_centroid_from_inner_mask(mask, depth_buffer, intr):
    # Convert tensor to numpy array if needed
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()

    # Find edges using Canny and then we use dilation operation to have a secure border which we exclude from the mask
    edges = cv2.Canny((mask.astype(np.uint8) * 255), threshold1=100, threshold2=200)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Inner mask can be found by rules: original mask has to be non-zero and location needs to be outside of border region
    inner_mask = (mask > 0) & (dilated_edges == 0)
    ys, xs = np.where(inner_mask)
    points_3d = []

    for x, y in zip(xs, ys):
        pixel_depths = [depth_frame[y, x] for depth_frame in depth_buffer]
        pixel_depths = [d for d in pixel_depths if d > 0]

        if not pixel_depths:
            continue

        median_depth_m = np.median(pixel_depths) # I found out by trial and error that the median works better (less prone to outliers)
        point_3d = rs.rs2_deproject_pixel_to_point(intr, [x, y], median_depth_m)
        points_3d.append(point_3d)

    if not points_3d:
        print("No valid 3D points found in inner mask")
        return None

    points_3d = np.array(points_3d)
    centroid = np.median(points_3d, axis=0)

    return centroid, inner_mask

@st.cache_resource
def load_groundingdino_model(repo_id, model_filename, config_filename, device='cpu'):
    import groundingdino.datasets.transforms as T  # type: ignore
    from groundingdino.models import build_model  # type: ignore
    from groundingdino.util.slconfig import SLConfig  # type: ignore
    from groundingdino.util.utils import clean_state_dict  # type: ignore
    from huggingface_hub import hf_hub_download
    """
    Load GroundingDINO model from Hugging Face Hub.
    """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=config_filename)
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=model_filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"GroundingDINO model loaded")
    model.eval()
    return model

@st.cache_resource
def load_sam_model(checkpoint_path, model_type="vit_b", device=torch.device('cpu')):
    from segment_anything import build_sam, SamPredictor, sam_model_registry
    """
    Load Segment Anything Model (SAM).
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"SAM model loaded")
    return predictor

@st.cache_resource
def load_stable_diffusion_inpaint(device=torch.device('cpu')):
    from diffusers import StableDiffusionInpaintPipeline

    """
    Load Stable Diffusion Inpaint pipeline.
    """
    if device.type == 'cpu':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch_dtype,
    )
    if device.type != 'cpu':
        pipe = pipe.to(device)
    print(f"Stable Diffusion Inpaint pipeline loaded on {device}")
    return pipe

CKPT_REPO_ID = "ShilongLiu/GroundingDINO"
GROUNDINGDINO_MODEL_FILENAME = "groundingdino_swinb_cogcoor.pth"
GROUNDINGDINO_CONFIG_FILENAME = "GroundingDINO_SwinB.cfg.py"

@st.cache_resource
def initialize_models(device=None, model_name='vit_b'):
    """
    Initialize and load all models.
    Returns a dictionary with models.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    groundingdino_model = load_groundingdino_model(
        CKPT_REPO_ID,
        GROUNDINGDINO_MODEL_FILENAME,
        GROUNDINGDINO_CONFIG_FILENAME,
        device=str(device)
    )

    # To choose between lightweight version or more heavy version. Results are 
    # little more detailed in vit_h, but prefer vit_b for speed and computation time
    if model_name == "vit_b":
        SAM_CHECKPOINT_PATH = "Vision/sam_vit_b_01ec64.pth"
        SAM_MODEL_TYPE = "vit_b"
    elif model_name == "vit_h":
        SAM_CHECKPOINT_PATH = "Vision/sam_vit_h_4b8939.pth"
        SAM_MODEL_TYPE = "vit_h"


    sam_predictor = load_sam_model(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, device)
    stable_diffusion_pipe = load_stable_diffusion_inpaint(device)

    return {
        "groundingdino_model": groundingdino_model,
        "sam_predictor": sam_predictor,
        "stable_diffusion_pipe": stable_diffusion_pipe,
        "device": device,
    }

def show_mask(mask, image, random_color=True):
    if random_color: # just use random color for every object to put as segmentation mask
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) #every 1 of mask becomes color
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA") #remember a is alpha (transparency)

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)) #Image.alpha_composite is code that can blend two images on top of eachother

def annotate_and_extract_centroids(indices, image_source, boxes, logits, phrases, sam_predictor, depth_buffer, depth_frame, device):
    from groundingdino.util.inference import annotate
    from groundingdino.util import box_ops

    selected_boxes = boxes[indices]
    selected_logits = logits[indices]
    selected_phrases = [phrases[i] for i in indices]

    annotated_frame = annotate(
        image_source=image_source,
        boxes=selected_boxes,
        logits=selected_logits,
        phrases=selected_phrases
    )
    cv2.imwrite("Vision/Scene_Images/annotated_output.jpg", annotated_frame)

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(selected_boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    centroids = []
    intr = depth_frame.profile.as_video_stream_profile().intrinsics

    for i in range(len(indices)):
        annotated_frame = show_mask(masks[i][0], annotated_frame)
        mask_np = masks[i][0].cpu().numpy()  # convert to numpy if needed
        mask_img = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(f"Vision/Masks/masks_{i}.png", mask_img)
        centroid, inner_mask = get_3d_centroid_from_inner_mask(masks[i][0], depth_buffer, intr)
        centroids.append(centroid)

    cv2.imwrite("Vision/Scene_Images/annotated_output_with_mask.jpg", annotated_frame)
    inner_mask_tensor = torch.from_numpy(inner_mask.astype(np.uint8))
    annotated_frame = show_mask(inner_mask_tensor, annotated_frame)
    cv2.imwrite("Vision/Scene_Images/annotated_output_with_inner_mask.jpg", annotated_frame)


    return centroids

def segment_object(text, image_str, intent, depth_buffer, depth_frame, fruits_of_interest):
    from groundingdino.util.inference import annotate, load_image, predict

    models = initialize_models()
    groundingdino = models["groundingdino_model"]
    sam_predictor = models["sam_predictor"]
    sd_pipe = models["stable_diffusion_pipe"]
    device = models["device"]

    image_source, image = load_image(image_str)

    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.40

    boxes, logits, phrases = predict(
        model=groundingdino, 
        image=image, 
        caption=text, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device=device
    )

    if len(phrases)==0:
        centroids = None
        F = None
        
    elif intent == 'single':
        I = [0]
        F = [phrases[0]]
        centroids = annotate_and_extract_centroids(I, image_source, boxes, logits, phrases, sam_predictor, depth_buffer, depth_frame, device)

    elif intent == 'multiple_1':
        F = []
        I = []
        indices_to_remove = []

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if j in indices_to_remove or i in indices_to_remove:
                    continue
                if torch.allclose(boxes[i], boxes[j], atol=5e-3): #to make sure if object had two labels, we only select highest
                    if logits[i] < logits[j]:
                        indices_to_remove.append(i)
                    else:
                        indices_to_remove.append(j)

        keep_indices = [i for i in range(len(boxes)) if i not in indices_to_remove]
        boxes = boxes[keep_indices]
        logits = logits[keep_indices]
        phrases = [phrases[i] for i in keep_indices]

        for i, phrase in enumerate(phrases):
            if phrase in fruits_of_interest:
                if phrase not in F:
                    F.append(phrase)
                    I.append(i)

        centroids = annotate_and_extract_centroids(I, image_source, boxes, logits, phrases, sam_predictor, depth_buffer, depth_frame, device)

    elif intent == 'multiple_2' or intent =='compare_all':
        I = []
        indices_to_remove = []
        print(phrases)


        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if j in indices_to_remove or i in indices_to_remove:
                    continue
                if torch.allclose(boxes[i], boxes[j], atol=5e-3):
                    if logits[i] < logits[j]:
                        indices_to_remove.append(i)
                    else:
                        indices_to_remove.append(j)
        
        keep_indices = [i for i in range(len(boxes)) if i not in indices_to_remove]
        boxes = boxes[keep_indices]
        logits = logits[keep_indices]
        phrases = [phrases[i] for i in keep_indices]
        F = phrases

        for i in range(len(boxes)):
            I.append(i)

        # for i, phrase in enumerate(phrases):
        #     if phrase in fruits_of_interest:
        #         if phrase not in F:
        #             F.append(phrase)
        #             I.append(i)

        centroids = annotate_and_extract_centroids(I, image_source, boxes, logits, phrases, sam_predictor, depth_buffer, depth_frame, device)

    return centroids, F 
        
        

    




