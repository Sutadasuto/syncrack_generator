import cv2
import numpy as np
import os

from shutil import copyfile


def compare_masks(gt_mask, pred_mask):
    new_image = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    new_image[..., 2][np.where(pred_mask >= 128)] = 255
    new_image[..., 0][np.where(gt_mask >= 128)] = 255
    new_image[..., 1][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 255
    new_image[..., 0][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 0
    new_image[..., 2][np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
    return new_image


def get_windows(img, window_size, sliding_steps):
    height, width = img.shape
    y_step = window_size[0] if sliding_steps[0] == -1 else sliding_steps[0]
    x_step = window_size[1] if sliding_steps[1] == -1 else sliding_steps[1]
    anchors = []
    x = y = 0
    while y < height:
        while x < width:
            anchors.append([y, x])
            x += x_step
        x = 0
        y += y_step

    windows = []
    for anchor in anchors:
        windows.append(img[anchor[0]: anchor[0] + window_size[0], anchor[1]: anchor[1] + window_size[1]])
    return windows, anchors


def join_windows(windows, anchors):
    window_height, window_width = windows[0].shape
    i = y = 0
    while True:
        last_width = windows[i].shape[1]
        i += 1
        y, x = anchors[i]
        if y > 0:
            width = anchors[i - 1][1] + last_width
            break
    height = anchors[-1][0] + windows[-1].shape[0]
    reconstructed = np.zeros((height, width), dtype=np.uint8)

    for window, anchor in zip(windows, anchors):
        reconstructed[anchor[0]: anchor[0] + window_height, anchor[1]: anchor[1] + window_width] = window
    return reconstructed


def attack_annotation(annotation, grid_percentage=0.2, dil_avg_size=3, er_avg_size=2, dilation_prob=0.95):
    image_size = annotation.shape
    window_height, window_width = int(image_size[0] * grid_percentage), int(image_size[1] * grid_percentage)
    windows, windows_anchors = get_windows(annotation, (window_height, window_width), (-1, -1))
    for window_index in range(len(windows)):
        operation = np.random.choice(["dilate", "erode"], p=[dilation_prob, 1 - dilation_prob])
        if operation == "dilate":
            dil_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                max(1, round(np.random.normal(dil_avg_size, 0.5))), max(1, round(np.random.normal(dil_avg_size, 0.5)))))
            windows[window_index] = cv2.dilate(windows[window_index], dil_se)
        else:
            er_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                max(1, round(np.random.normal(er_avg_size, 0.5))), max(1, round(np.random.normal(er_avg_size, 0.5)))))
            windows[window_index] = cv2.erode(windows[window_index], er_se)

    attacked_annotation = join_windows(windows, windows_anchors)
    attacked_annotation = cv2.morphologyEx(attacked_annotation, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_avg_size, dil_avg_size)),
                                           iterations=1)
    attacked_annotation = cv2.morphologyEx(attacked_annotation, cv2.MORPH_OPEN,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er_avg_size, er_avg_size)),
                                           iterations=1)
    return attacked_annotation


def attack_dataset(path_to_dataset):
    if not os.path.exists(path_to_dataset + "_attacked"):
        os.makedirs(path_to_dataset + "_attacked")
    if not os.path.exists(path_to_dataset + "_label_comparison"):
        os.makedirs(path_to_dataset + "_label_comparison")
    string_of_interest = "_gt.png"

    ground_truth_image_paths = sorted(
        [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset)
         if not f.startswith(".") and f.endswith(string_of_interest)],
        key=lambda f: f.lower())

    print("Attacking ground truths ({:.0f}%)...".format(0), end="\r")
    for i, gt_path in enumerate(ground_truth_image_paths):
        new_gt_path = os.path.join(path_to_dataset + "_attacked", os.path.split(gt_path)[1])
        comparison_path = os.path.join(path_to_dataset + "_label_comparison", os.path.split(gt_path)[1])
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_attacked = attack_annotation(gt)
        img_path = gt_path.replace(string_of_interest, ".jpg")
        new_img_path = os.path.join(path_to_dataset + "_attacked", os.path.split(img_path)[1])
        copyfile(img_path, new_img_path)
        cv2.imwrite(new_gt_path, gt_attacked)
        gt_comparison = compare_masks(gt, gt_attacked)
        cv2.imwrite(comparison_path,
                    np.concatenate(
                        [
                            np.concatenate([gt[..., None] for c in range(3)], axis=-1),
                            np.concatenate([gt_attacked[..., None] for c in range(3)], axis=-1),
                            gt_comparison
                        ], axis=1)
                    )
        print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)), end="\r")
    print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)))


# attack_dataset("/media/winbuntu/databases/syncrack_dataset")
