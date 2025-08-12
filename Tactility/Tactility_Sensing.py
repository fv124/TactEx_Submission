# 0. Define Packages
import time
import os
import random
import csv
from xarm.wrapper import XArmAPI
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 1. Define Contact Criteria: see Collect_Data for more explanation or paper
#Based on two criteria: 1) marker displacement should be >=2, 2) SSIM should be <0.96
# The combination minimizes false contacts while being sensitive to soft materials
def label_markers(image_path, plot=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    h, w = binary.shape
    valid_centroids = []

    for i in range(1, num_labels):  # Skip background, therefore start from 1
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]
        cx, cy = int(cx), int(cy)

        if 10 < area < 750 and 75 < x < w - 75 and 75 < y < h - 75:
            new_centroid = np.array([cx, cy])
            merged = False

            for j, existing in enumerate(valid_centroids):
                if np.linalg.norm(existing - new_centroid) < 20:
                    valid_centroids[j] = (existing + new_centroid) / 2
                    merged = True

            if not merged:
                if len(valid_centroids) != 63: #because I noticed right under has black dot possibly that should not be in there
                    valid_centroids.append(new_centroid)

    # Convert to int tuples for drawing
    valid_centroids = [tuple(map(int, c)) for c in valid_centroids]

    if plot:
        img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in valid_centroids:
            cv2.circle(img_debug, (cx, cy), 4, (0, 255, 0), -1)

        plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected markers: {len(valid_centroids)}")
        plt.axis("off")
        plt.show()

    return np.array(valid_centroids)

def match_unique_pairs(centroids_ref, centroids_def, max_distance=40):
    # Compute pairwise distance matrix
    dist_matrix = cdist(centroids_def, centroids_ref)

    # Mask out impossible matches
    dist_matrix[dist_matrix > max_distance] = 100

    # Solve assignment: linear_sum_assignment can find optimal way to assign the points while minimizing costs
    row_idx, col_idx = linear_sum_assignment(dist_matrix)

    # Filter by max distance again (in case no match was acceptable)
    valid = dist_matrix[row_idx, col_idx] < max_distance

    matched_def = centroids_def[row_idx[valid]]
    matched_ref = centroids_ref[col_idx[valid]]

    return matched_ref, matched_def

def calculate_marker_displacement(image_path1, image_path2):
    centroids_ref = label_markers(image_path1)
    centroids_def = label_markers(image_path2)

    matched_ref, matched_def = match_unique_pairs(centroids_ref, centroids_def)

    displacement_vectors = matched_def - matched_ref
    magnitudes = np.linalg.norm(displacement_vectors, axis=1)

    return np.mean(magnitudes)

def calculate_color_ssim(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Compute SSIM with smaller win_size for small images
    score, diff = ssim(img1, img2, multichannel=True, full=True, win_size=3)
    return score, diff

# 2. Set record and save_image
def record_and_save_image(path):
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (512, 512))
    cv2.imwrite(path, frame)

# 3. Make Contact
def make_contact(object, arm, path="Tactility/Data", HA=0, N_Data=1):
    pose = arm.get_position()[1]
    with open(os.path.join(path, "Data_Overview.csv"), 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(N_Data):
            status = "Not Touched"
            z_change = pose[2]
            ref_path = os.path.join(path, f'{object}_{HA}_Ref_{i}.png')
            record_and_save_image(ref_path)
            s = 0
            writer.writerow([object, i, s, HA, 'Reference', 'Reference', z_change])

            while status != 'Done':
                if status == 'Not Touched':
                    z_change-= 1
                    arm.set_position(z=z_change)

                    contact1_path = os.path.join(path, f'{object}_{HA}_Contact1_{i}.png')
                    record_and_save_image(contact1_path)
                    time.sleep(0.25)

                    contact_score = calculate_color_ssim(ref_path, contact1_path)[0]
                    marker_displacement = calculate_marker_displacement(ref_path, contact1_path)
                    if contact_score <= 0.96 and marker_displacement >= 2:
                        print('Contact Detected')
                        print(f"SSIM: {contact_score:.4f}")
                        z1 = z_change
                        s = 1
                        writer.writerow([object, i, s, HA, contact_score, marker_displacement, z_change])
                        status = "Touched"
                if status == 'Touched':
                    s += 1
                    z_change -= 0.25
                    arm.set_position(z=z_change)

                    contact_path = os.path.join(path, f'{object}_{HA}_Contact{s}_{i}.png')
                    record_and_save_image(contact_path)
                    time.sleep(0.25)
                    
                    contact_score = calculate_color_ssim(ref_path, contact_path)[0]
                    marker_displacement = calculate_marker_displacement(ref_path, contact_path)
                    print(f"SSIM: {contact_score:.4f}")
                    print(f"Marker Displacement: {marker_displacement:.4f}")
                    writer.writerow([object, i, s, HA, contact_score, marker_displacement, z_change])
                    
                    if s == 8:
                        status = 'Done'
                        print('Done, Resetting...')
                        arm.set_position(z=pose[2])
                        time.sleep(1)

                    
    arm.set_position(x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5])

