import pyrealsense2 as rs
import cv2
import numpy as np

def get_filtered_depth(depth_frame):
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    return depth_frame.as_depth_frame()

def get_buffer_images():
    # === RealSense setup ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_buffer = []

    for _ in range(10):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        
        # Apply filters here if you want (spatial, temporal, hole filling)
        depth_frame = get_filtered_depth(depth_frame)

        # Convert to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        depth_buffer.append(depth_image)
        
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite("Vision/Scene_Images/captured_image.jpg", color_image)
    pipeline.stop()

    return depth_buffer, depth_frame