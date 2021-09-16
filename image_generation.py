import noise
import numpy as np
import cv2

from copy import deepcopy
from scipy import ndimage
from noisy_labels import get_windows, join_windows


def get_pavement(input_size, octaves=5, scale=3.5, persistence=1, lacunarity=2.0, base=0):
    width, height = input_size
    canvas = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            canvas[i][j] = noise.pnoise2(i / 2**scale, j / (2 ** scale), octaves=octaves, persistence=persistence,
                                         lacunarity=lacunarity, repeatx=width, repeaty=height, base=base)
    return (canvas + 1) / 2


def colorize_pavement(canvas, dark_color=(56, 46, 46), bright_color=(184, 178, 178)):
    bgr_canvas = np.concatenate([canvas[:, :, None] for i in range(3)], axis=2)
    bgr_canvas[:, :, 0] = (1 - canvas) * dark_color[0] + (canvas) * bright_color[0]
    bgr_canvas[:, :, 1] = (1 - canvas) * dark_color[1] + (canvas) * bright_color[1]
    bgr_canvas[:, :, 2] = (1 - canvas) * dark_color[2] + (canvas) * bright_color[2]
    return bgr_canvas


def add_salt_pepper(colorized_pavement, amount=0.002, s_vs_p=0.7, mean_rad=1, std_rad=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    def get_axis(default_rng):
        return int(round(max(0, round(default_rng.normal(mean_rad, std_rad)))))

    out = np.copy(colorized_pavement)
    # Salt mode
    num_salt = int(np.ceil(amount * colorized_pavement.size * s_vs_p))
    salt_color = np.array([colorized_pavement.max() for channel in range(colorized_pavement.shape[-1])])
    coords = [rng.integers(0, i - 1, num_salt)
              for i in colorized_pavement.shape[:-1]]
    for grain in range(len(coords[0])):
        if get_axis(rng) > 0:
            out = cv2.ellipse(out, (coords[1][grain], coords[0][grain]), (max(1, get_axis(rng)), max(1, get_axis(rng))),
                              angle=rng.normal(90, 22), startAngle=0, endAngle=rng.choice([360, 270, 200]),
                              color=salt_color, thickness=-1)
        else:
            out[coords[0][grain], coords[1][grain]] = salt_color


    # Pepper mode
    num_pepper = int(np.ceil(amount * colorized_pavement.size * (1. - s_vs_p)))
    pepper_color = np.array([colorized_pavement.min() for channel in range(colorized_pavement.shape[-1])])
    coords = [rng.integers(0, i - 1, num_pepper)
              for i in colorized_pavement.shape[:-1]]
    for grain in range(len(coords[0])):
        if get_axis(rng) > 0:
            out = cv2.ellipse(out, (coords[1][grain], coords[0][grain]), (max(1, get_axis(rng)), max(1, get_axis(rng))),
                              angle=rng.normal(90, 25), startAngle=0, endAngle=rng.choice([360, 270, 200]),
                              color=pepper_color, thickness=-1)
        else:
            out[coords[0][grain], coords[1][grain]] = pepper_color

    return 0.7*out + 0.3*colorized_pavement


def add_cracky_noise(colorized_pavement, intensity=0.7, amount=0.002, mean_rad=2, std_rad=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
        
    def get_axis(default_rng):
        return int(round(max(0, round(default_rng.normal(mean_rad, std_rad)))))

    noisy_pavement = np.copy(colorized_pavement)

    mask = np.ones(colorized_pavement.shape[:-1], dtype=np.float)
    num_cracks = int(np.ceil(amount * colorized_pavement[..., 0].size))
    coords = [rng.integers(0, i - 1, num_cracks)
              for i in colorized_pavement.shape[:-1]]

    for grain in range(len(coords[0])):
        if get_axis(rng) > 0:
            start_angle = rng.normal(90, 22)
            end_angle = rng.normal(start_angle + 90, 22)
            mask = cv2.ellipse(mask, (coords[1][grain], coords[0][grain]),
                               (max(1, get_axis(rng)), max(1, get_axis(rng))),
                               angle=rng.normal(90, 22), startAngle=start_angle, endAngle=end_angle,
                               color=min(max(0.0, rng.normal(intensity, intensity/10)), 0.9999),
                               thickness=1)
        else:
            mask[coords[0][grain], coords[1][grain]] = min(max(0.0, rng.normal(intensity, intensity/10)), 0.9999)

    mask = cv2.blur(mask, (2,2))
    for channel in range(3):
        noisy_pavement[..., channel] *= mask

    return noisy_pavement


def oriented_gradient_fill(image_size, slope=1, center=0.5):
    canvas = np.zeros(image_size, np.float)
    slope *= -1
    y_cen = int(center * image_size[0])
    x_cen = int(center * image_size[1])
    color_step_size = 0.5/x_cen
    x_2_0 = int(-(y_cen + x_cen) / slope)
    x_1_0 = int(((image_size[1] - 1) - (y_cen + x_cen)) / slope)

    color = 0.5
    for x in range(max(x_1_0, x_2_0)):
        canvas = cv2.line(canvas, (x_1_0 - x, image_size[0] - 1), (x_2_0 - x, 0), color=max(0.0, color), thickness=1)
        color -= color_step_size
    color = 0.5
    for x in range(image_size[1] - min(x_1_0, x_2_0)):
        canvas = cv2.line(canvas, (x_1_0 + x, image_size[0] - 1), (x_2_0 + x, 0), color=min(1.0, color), thickness=1)
        color += color_step_size
    return cv2.medianBlur(canvas.astype(np.float32), 5)


def create_crack_shape(bounding_length, bounding_width, octaves=8, scale=4, persistence=0.5, lacunarity=2,
                       crack_avg_width=2, base=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.array([round(i * scale) for i in range(int(bounding_length/scale))], dtype=np.int)
    y = np.zeros(x.shape, dtype=np.int)
    canvas = np.zeros((bounding_width, bounding_length), np.uint8)
    for i in range(len(x)):
        y[i] = int(
            # 0.5 * bounding_width * (1 + noise.pnoise1(i / 2**scale, octaves=octaves, persistence=persistence,
            0.5 * bounding_width * (1 + noise.pnoise1(i/len(x), octaves=octaves, persistence=persistence,
                                                      lacunarity=lacunarity, base=base, repeat=bounding_length)))
        if i > 0:
            canvas = cv2.line(canvas, (x[i - 1], y[i - 1]), (x[i], y[i]), color=255, thickness=1)
    er_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (crack_avg_width, crack_avg_width))
    canvas = cv2.dilate(canvas, er_se)
    if crack_avg_width > 1:
        window_height, window_width = int(canvas.shape[0]), int(canvas.shape[1] * 0.03)
        windows, windows_anchors = get_windows(canvas, (window_height, window_width), (-1, -1))
        erode_prob = 0.4
        for window_index in range(len(windows)):
            operation = rng.choice(["erode", "nothing"], p=[erode_prob, 1 - erode_prob])
            if operation == "erode":
                er_se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
                windows[window_index] = cv2.erode(windows[window_index], er_se)
        # canvas = join_windows(windows, windows_anchors)
        # window_height, window_width = int(canvas.shape[0]), int(canvas.shape[1] * 0.05)
        # windows, windows_anchors = get_windows(canvas, (window_height, window_width), (-1, -1))
        # er_se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, crack_width))
        # windows[0] = cv2.erode(windows[0], er_se)
        # windows[-1] = cv2.erode(windows[1], er_se)
        canvas = join_windows(windows, windows_anchors)
    return canvas


def add_crack(pavement_image, crack_mask, intensity_avg=0.75, fade_factor=0.95, position=(0, 0), orientation=0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if orientation != 0:
        crack_mask = cv2.morphologyEx(ndimage.rotate(crack_mask, orientation), cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    ret, crack_mask = cv2.threshold(crack_mask, 100, 1.0, cv2.THRESH_BINARY)
    # crack_core_mask = cv2.erode(crack_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #
    # window_height, window_width = int(crack_mask.shape[0] * 0.10), int(crack_mask.shape[1] * 0.10)
    # windows, windows_anchors = get_windows(crack_mask, (window_height, window_width), (-1, -1))
    # for window_index in range(len(windows)):
    #     se_size = rng.choice([1, 3, 5], p=[0.10, 0.60, 0.30])
    #     windows[window_index] = cv2.dilate(windows[window_index], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size)))
    # crack_surround_mask = join_windows(windows, windows_anchors)
    # crack_surround_mask = cv2.bitwise_xor(crack_mask, crack_surround_mask)
    #
    # surrounding_pixels = np.where(crack_surround_mask == 1.0)
    # mask = np.ones(pavement_image.shape)[..., 0]
    # for pixel_num in range(len(surrounding_pixels[0])):
    #     try:
    #         mask[position[1] + surrounding_pixels[0][pixel_num], position[0] + surrounding_pixels[1][pixel_num]] = \
    #             min(max(intensity_avg / (fade_factor ** 4), rng.normal(intensity_avg / (fade_factor ** 4), 0.10)), 1.0)
    #     except IndexError:
    #         continue
    # pavement_image = np.copy(pavement_image)
    # for channel in range(3):
    #     pavement_image[..., channel] *= mask

    mask = np.ones(pavement_image.shape)[..., 0]

    crack_pixels = np.where(crack_mask == 1.0)
    for pixel_num in range(len(crack_pixels[0])):
        try:
            mask[position[1] + crack_pixels[0][pixel_num], position[0] + crack_pixels[1][pixel_num]] = \
                min(max(0.0, rng.normal(intensity_avg / (fade_factor), intensity_avg / 10)), 0.9999)
        except IndexError:
            continue

    # crack_core_pixels = np.where(crack_core_mask == 1.0)
    # for pixel_num in range(len(crack_core_pixels[0])):
    #     try:
    #         mask[position[1] + crack_core_pixels[0][pixel_num], position[0] + crack_core_pixels[1][pixel_num]] =\
    #             min(max(0.0, rng.normal(intensity_avg, intensity_avg / 10)), 0.9999)
    #     except IndexError:
    #         continue

    # Change the intensity distribution of cracks along the image (a crack has not the same intensity in all the structure)
    compensated_masks = []
    compensation_iterations = 2
    for d in range(1, compensation_iterations + 1):
        compensated_mask = np.zeros(mask.shape, dtype=np.float)
        compensated_mask[np.where(mask < 1.0)] = mask[np.where(mask < 1.0)]
        window_height, window_width = int(compensated_mask.shape[0] * 0.25/d), int(compensated_mask.shape[1] * 0.25/d)
        windows, windows_anchors = get_windows(compensated_mask, (window_height, window_width), (-1, -1))
        for window_index in range(len(windows)):
            intensity_change = rng.choice([1.0, 0.9, 1.1])
            windows[window_index] = np.minimum(0.90, windows[window_index] * intensity_change)
        compensated_masks.append(join_windows(windows, windows_anchors))
    mask = np.average(np.array(compensated_masks), axis=0)
    mask[np.where(mask == 0.0)] = 1.0

    window_height, window_width = int(mask.shape[0] * 0.10), int(mask.shape[1] * 0.10)
    windows, windows_anchors = get_windows(mask, (window_height, window_width), (-1, -1))
    secondary_windows = deepcopy(windows)
    for window_index in range(len(windows)):
        se_size = rng.choice([i for i in range(2, 6)])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        windows[window_index] = cv2.filter2D(windows[window_index], -1, kernel/np.sum(kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*se_size, 2*se_size))
        secondary_windows[window_index] = cv2.filter2D(secondary_windows[window_index], -1, kernel / np.sum(kernel))
    crack_surround_mask = join_windows(windows, windows_anchors)
    secondary_crack_surround_mask = join_windows(secondary_windows, windows_anchors)
    crack_surround_mask = np.minimum(crack_surround_mask, secondary_crack_surround_mask)

    inserted_crack_mask = np.copy(crack_surround_mask)
    ret, gt = cv2.threshold(mask, 0.99991, 1.0, cv2.THRESH_BINARY_INV)
    inserted_crack_mask[np.where(gt == 1.0)] = mask[np.where(gt == 1.0)]

    pavement_image = np.copy(pavement_image)
    for channel in range(3):
        pavement_image[..., channel] *= inserted_crack_mask

    return pavement_image.astype(np.uint8), (255*gt).astype(np.uint8)
