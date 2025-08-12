import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

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

    return centroid

def segment_from_YOLO(image_str, fruits_of_interest, depth_buffer, depth_frame):
    model = YOLO("Vision/YOLO_Trained.pt")
    img = cv2.imread(image_str)

    results = model(img)
    img_annotated = results[0].plot()
    cv2.imwrite("Vision/Scene_Images/YOLO_Annotated_with_mask.jpg", cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))

    centroids = []
    F = []
    for i in range(len(results[0].masks)):
        cl = int(results[0].boxes.cls[i])
        if (model.names[cl].lower() in fruits_of_interest) and (results[0].boxes.conf[i] > 0.60):
            F.append(model.names[cl].lower())

            mask_tensor = results[0].masks.data[i]
            mask_np = cv2.resize(mask_tensor.numpy().astype(np.uint8), (img_annotated.shape[1], img_annotated.shape[0]), interpolation=cv2.INTER_NEAREST)
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            centroid = get_3d_centroid_from_inner_mask(mask_np, depth_buffer, intr)
            centroids.append(centroid)

    # results.masks is a list or tensor of masks, one per detected object
    for i, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy()  
        mask_img = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(f"Vision/Masks/mask_{i}.png", mask_img)
    
    return centroids, F

