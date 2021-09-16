import cv2
import numpy as np
import os

from shutil import copyfile


def compare_masks(gt_mask, pred_mask, bg_color="white"):
    if bg_color == "black":
        new_image = np.zeros(gt_mask.shape, dtype=np.float32)
        new_image[..., 2][np.where(pred_mask[..., 0] >= 128)] = 255
        new_image[..., 0][np.where(gt_mask[..., 0] >= 128)] = 255
        new_image[..., 1][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 255
        new_image[..., 0][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 0
        new_image[..., 2][
            np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
    else:
        new_image = 255 * np.ones(gt_mask.shape, dtype=np.float32)
        new_image[..., 0][np.where(pred_mask[..., 0] < 128)] = 0
        new_image[..., 2][np.where(gt_mask[..., 0] < 128)] = 0
        new_image[..., 1][
            np.where((new_image[..., 0]) == 0 & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
        new_image[..., 1][
            np.where((new_image[..., 2]) == 0 & (new_image[..., 0] == 255) & (new_image[..., 1] == 255))] = 0
        new_image[..., 1][
            np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 0) & (new_image[..., 2] == 0))] = 255

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
        if i == len(anchors):
            width = anchors[i - 1][1] + last_width
            break
        y, x = anchors[i]
        if y > 0:
            width = anchors[i - 1][1] + last_width
            break
    height = anchors[-1][0] + windows[-1].shape[0]
    reconstructed = np.zeros((height, width), dtype=windows[0].dtype)

    for window, anchor in zip(windows, anchors):
        reconstructed[anchor[0]: anchor[0] + window_height, anchor[1]: anchor[1] + window_width] = window
    return reconstructed


def attack_annotation(annotation, operation='both', noise_prob=1.0, grid_percentage=0.1, dil_avg_size=2, er_avg_size=2,
                      dilation_prob=0.50):

    noise_prob *= 100
    image_size = annotation.shape
    window_height, window_width = int(image_size[0] * grid_percentage), int(image_size[1] * grid_percentage)
    windows, windows_anchors = get_windows(annotation, (window_height, window_width), (-1, -1))
    for window_index in range(len(windows)):
        if np.random.choice(100) > noise_prob:
            continue
        if operation is "both":
            current_operation = np.random.choice(["dilate", "erode"], p=[dilation_prob, 1 - dilation_prob])
        else:
            current_operation = operation
        if current_operation == "dilate":
            se_size = max(2, round(np.random.normal(dil_avg_size, 0.5)))
            if se_size % 2 == 0:
                se_size += 1
            dil_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
            windows[window_index] = cv2.dilate(windows[window_index], dil_se,
                                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            se_size = max(2, round(np.random.normal(er_avg_size, 0.5)))
            if se_size % 2 == 0:
                se_size += 1
            er_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
            windows[window_index] = cv2.erode(windows[window_index], er_se,
                                               borderType=cv2.BORDER_CONSTANT, borderValue=0)

    attacked_annotation = join_windows(windows, windows_anchors)
    return attacked_annotation


# def attack_annotation_shift(annotation, grid_percentage=0.1):
#     image_size = annotation.shape
#     window_height, window_width = int(image_size[0] * grid_percentage), int(image_size[1] * grid_percentage)
#     windows, windows_anchors = get_windows(annotation, (window_height, window_width), (-1, -1))
#     shifts = [i for i in range(-4, 5)]
#     for window_index in range(len(windows)):
#         operation = np.random.choice(["vertical", "horizontal"], p=[0.5, 0.5])
#         shift = np.random.choice(shifts)
#         if operation == "vertical":
#             M = np.float32([[1, 0, 0], [0, 1, shift]])
#         else:
#             M = np.float32([[1, 0, shift], [0, 1, 0]])
#
#         windows[window_index] = cv2.warpAffine(windows[window_index], M, (window_width, window_height))
#
#     attacked_annotation = join_windows(windows, windows_anchors)
#     return attacked_annotation


def attack_dataset(path_to_dataset, bg_color="white"):
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

        if bg_color == "white":
            gt = 255 - gt
            gt_attacked = 255 - gt_attacked

        gt = np.concatenate([gt[..., None] for c in range(3)], axis=-1)
        gt_attacked = np.concatenate([gt_attacked[..., None] for c in range(3)], axis=-1)
        gt_comparison = compare_masks(gt, gt_attacked)
        cv2.imwrite(comparison_path, np.concatenate([gt, gt_attacked, gt_comparison], axis=1))
        print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)), end="\r")
    print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)))


def compare_gt_stats(original_path, path_to_compare, destination_path=None):

    if destination_path is None:
        destination_path = os.path.split(original_path)[0]
    destination_name = "%s-VS-%s.csv" % (os.path.split(original_path)[1], os.path.split(path_to_compare)[1])

    image_names = sorted([f for f in os.listdir(original_path)
                          if not f.startswith(".") and f.endswith("_gt.png")],
                         key=lambda f: f.lower())
    dir_name = os.path.split(path_to_compare)[-1]

    with open(os.path.join(destination_path, destination_name), "w+") as f:
        string_list = ["Image,TP,FP,TN,FN,Total pixels"]
        t_tp = 0
        t_fp = 0
        t_fn = 0
        t_tn = 0
        t_n_pix = 0
        for name in image_names:
            or_gt = cv2.imread(os.path.join(original_path, name))
            ex_gt = cv2.imread(os.path.join(path_to_compare, name))

            comparative_mask = (compare_masks(or_gt, ex_gt, "black") / 255)
            n_pix = comparative_mask.shape[0] * comparative_mask.shape[1]
            t_n_pix += n_pix
            tp = int((comparative_mask[..., 1]).sum())
            t_tp += tp
            fp = int((comparative_mask[..., 2]).sum())
            t_fp += fp
            fn = int((comparative_mask[..., 0]).sum())
            t_fn += fn
            tn = len(np.where(cv2.cvtColor(comparative_mask, cv2.COLOR_BGR2GRAY) == 0)[0])
            t_tn += tn

            string_list.append("%s,%s,%s,%s,%s,%s" % (os.path.join(dir_name, name), tp, fp, tn, fn, n_pix))
        string_list.append("%s,%s,%s,%s,%s,%s" % ('Total', t_tp, t_fp, t_tn, t_fn, t_n_pix))
        string_list.append("{},{:.2f},{:.2f},{:.2f},{:.2f},{}".format('Percentages', 100*t_tp/t_n_pix, 100*t_fp/t_n_pix,
                                                                     100*t_tn/t_n_pix, 100*t_fn/t_n_pix,
                                                                     100*t_n_pix/t_n_pix))
        f.write("\n".join(string_list))
