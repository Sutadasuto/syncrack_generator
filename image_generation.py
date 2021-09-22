import noise
import numpy as np
import cv2

from copy import deepcopy
from scipy import ndimage
from noisy_labels import get_windows, join_windows


# Create a pavement-like texture based on Perlin noise
def get_pavement(input_size, octaves=5, scale=3.5, persistence=1, lacunarity=2.0, base=0):
    width, height = input_size
    canvas = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            canvas[i][j] = noise.pnoise2(i / 2 ** scale, j / (2 ** scale), octaves=octaves, persistence=persistence,
                                         lacunarity=lacunarity, repeatx=width, repeaty=height, base=base)
    return (canvas + 1) / 2


# Given two colors, and a Perlin based texture, create a color version of the texture
def colorize_pavement(canvas, dark_color=(56, 46, 46), bright_color=(184, 178, 178)):
    bgr_canvas = np.concatenate([canvas[:, :, None] for i in range(3)], axis=2)
    bgr_canvas[:, :, 0] = (1 - canvas) * dark_color[0] + (canvas) * bright_color[0]
    bgr_canvas[:, :, 1] = (1 - canvas) * dark_color[1] + (canvas) * bright_color[1]
    bgr_canvas[:, :, 2] = (1 - canvas) * dark_color[2] + (canvas) * bright_color[2]
    return bgr_canvas


# Add salt&pepper noise to an image
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

    return 0.7 * out + 0.3 * colorized_pavement


# Add small artifacts to a background image. These will look similar to cracks, to make crack detection harder
def add_cracky_noise(colorized_pavement, intensity=0.7, amount=0.002, mean_rad=2, std_rad=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    def get_axis(default_rng):
        return int(round(max(0, round(default_rng.normal(mean_rad, std_rad)))))

    noisy_pavement = np.copy(colorized_pavement)

    mask = np.ones(colorized_pavement.shape[:-1], dtype=np.float)  # Mask of artifacts
    num_cracks = int(np.ceil(amount * colorized_pavement[..., 0].size))  # Number of artifacts
    coords = [rng.integers(0, i - 1, num_cracks)
              for i in colorized_pavement.shape[:-1]]  # Get artifact position

    for grain in range(len(coords[0])):
        if get_axis(rng) > 0:  # Artifact is not circular (e.g. it is elliptical)
            start_angle = rng.normal(90, 22)
            end_angle = rng.normal(start_angle + 90, 22)
            # Draw a semi-ellipsis into the mask (a small crack-like chape) in the given position
            mask = cv2.ellipse(mask, (coords[1][grain], coords[0][grain]),
                               (max(1, get_axis(rng)), max(1, get_axis(rng))),
                               angle=rng.normal(90, 22), startAngle=start_angle, endAngle=end_angle,
                               color=min(max(0.0, rng.normal(intensity, intensity / 10)), 0.9999),
                               thickness=1)
        else:  # Draw simply a point
            mask[coords[0][grain], coords[1][grain]] = min(max(0.0, rng.normal(intensity, intensity / 10)), 0.9999)

    # Blur the artifacts to have transition between artifact and background (just as with cracks)
    mask = cv2.blur(mask, (2, 2))
    # Introduce the artifacts in the image
    for channel in range(3):
        noisy_pavement[..., channel] *= mask

    return noisy_pavement


# Calculate alphas for two colors along an image to create a two-colors gradient)
def oriented_gradient_fill(image_size, slope=1, center=0.5):
    canvas = np.zeros(image_size, np.float)
    slope *= -1  # Since the vertical axis increases downwards in images
    y_cen = int(center * image_size[0])
    x_cen = int(center * image_size[1])
    alpha_step_size = 0.5 / x_cen
    # Given a known point of the center line, and it's slope, calculate the x-coordinates of the extreme points
    x_2_0 = int(-(y_cen + x_cen) / slope)
    x_1_0 = int(((image_size[1] - 1) - (y_cen + x_cen)) / slope)

    # Calculate alphas for color 1
    alpha = 0.5
    for x in range(max(x_1_0, x_2_0)):
        # The alphas are assigned line-wise with lines parallel to the input slope
        canvas = cv2.line(canvas, (x_1_0 - x, image_size[0] - 1), (x_2_0 - x, 0), color=max(0.0, alpha), thickness=1)
        alpha -= alpha_step_size  # As move along the image, alpha changes
    # Calculate alphas for color 2
    alpha = 0.5
    for x in range(image_size[1] - min(x_1_0, x_2_0)):
        canvas = cv2.line(canvas, (x_1_0 + x, image_size[0] - 1), (x_2_0 + x, 0), color=min(1.0, alpha), thickness=1)
        alpha += alpha_step_size
    return cv2.medianBlur(canvas.astype(np.float32), 5)


# Generate a mask of a crack. The mask size is the same as the minimum bounding box surrounding the crack with a 1-pixel
# margin. Crack is white, background is blac
def create_crack_shape(bounding_length, bounding_width, r_x, octaves=8, scale=4, persistence=0.5, lacunarity=2,
                       crack_avg_width=2, base=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    # Get the x-coordinates of the Perlin curve vertices
    x = np.array([round(i * scale) for i in range(int(bounding_length / scale))], dtype=np.int)
    y = np.zeros(x.shape, dtype=np.float)
    canvas = np.zeros((bounding_length, bounding_length), np.uint8)
    # Get the y-coordinates of the Perlin curve vertices
    for i in range(len(x)):
        y[i] = noise.pnoise1(i / len(x), octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base,
                             repeat=bounding_length)
    y = np.array(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize the points to [0, 1]
    y = (y * bounding_width)  # Map the points to the bounding width

    # Displace the points to follow an elliptic center line
    for i, value in enumerate(x):
        y[i] = y[i] + bounding_length * (1 - (r_x ** 2 - value ** 2) ** 0.5 / r_x)
    displacement = int((bounding_length - (np.max(y) - np.min(y))) / 2)

    # Translate the points to the center of the canvas
    y = (y + displacement).astype(np.int)
    for i in range(1, len(x)):
        canvas = cv2.line(canvas, (x[i - 1], y[i - 1]), (x[i], y[i]), color=255, thickness=1)

    # Provide thickness to the crack shape
    if crack_avg_width > 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (crack_avg_width, crack_avg_width))
        canvas = cv2.dilate(canvas, se)

    # Since real cracks don't have a fixed width, modify the width at certain regions
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    window_height, window_width = int(canvas.shape[0]), int(canvas.shape[1] * 0.03)
    windows, windows_anchors = get_windows(canvas, (window_height, window_width), (-1, -1))
    morph_prob = 0.5  # Probability of changing the width
    dilate_prob = 0.5  # Probability of making the crack wider if the width is changed

    operations = np.array(["nothing", "dilate", "erode"])
    prev_op = "nothing"
    for window_index in range(len(windows)):
        if prev_op == "nothing":
            operation = rng.choice(operations,
                                   p=[morph_prob, (1 - morph_prob) * dilate_prob, (1 - morph_prob) * (1 - dilate_prob)])
        else:  # Original case; a dilation followed by an erosion (and vice versa) looks bad
            operation = rng.choice(operations[operations != prev_op],
                                   p=[morph_prob, (1 - morph_prob)])
        prev_op = operation
        if operation == "erode":
            if crack_avg_width > 1:  # i.e. The crack won't disappear after the erosion
                windows[window_index] = cv2.erode(windows[window_index], se)
        elif operation == "dilate":
            windows[window_index] = cv2.dilate(windows[window_index], se)
    canvas = join_windows(windows, windows_anchors)

    # Crop crack image to the minimum bounding box with 1-pixel width borders
    x, y, w, h = cv2.boundingRect(canvas)
    canvas = canvas[y:y + h, x:x + w]
    output = np.zeros((canvas.shape[0] + 2, canvas.shape[1] + 2), dtype=np.uint8)
    output[1:1 + h, 1:1 + w] = canvas
    return output


def add_crack(pavement_image, crack_mask, intensity_avg=0.75, fade_factor=0.95, orientation=0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if orientation != 0:  # Rotate the crack
        # The mask undergoes a closing operation since the deformation caused by rotating can create holes in the crack
        crack_mask = cv2.morphologyEx(ndimage.rotate(crack_mask, orientation), cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    ret, crack_mask = cv2.threshold(crack_mask, 100, 1.0, cv2.THRESH_BINARY)  # Get binary mask

    # Crop crack image to the minimum bounding box
    x, y, w, h = cv2.boundingRect(crack_mask)
    crack_mask = crack_mask[y:y + h, x:x + w]

    # Choose a random position to introduce the crack into the pavement image
    position = (
        rng.integers(0, max(1, pavement_image.shape[0] - crack_mask.shape[0])),
        rng.integers(0, max(1, pavement_image.shape[1] - crack_mask.shape[1]))
    )

    # This mask will save the percentage of intensity to assign to each pixel in the pavement image
    mask = np.ones(pavement_image.shape)[..., 0]

    crack_pixels = np.where(crack_mask == 1.0)
    # To all the pixels labeled as crack, assign an intensity lower than 1 (calculated randomly based on the intensity
    # average provided as function input)
    for pixel_num in range(len(crack_pixels[0])):
        try:
            mask[position[0] + crack_pixels[0][pixel_num], position[1] + crack_pixels[1][pixel_num]] = \
                min(max(0.0, rng.normal(intensity_avg / (fade_factor), intensity_avg / 10)), 0.9999)
        except IndexError:
            continue

    # Change the intensity distribution of cracks along the image (using a sliding window approach).
    # A crack shouldn't the same intensity in all the structure.
    compensated_masks = []
    compensation_iterations = 2
    for d in range(1, compensation_iterations + 1):
        compensated_mask = np.zeros(mask.shape, dtype=np.float)
        compensated_mask[np.where(mask < 1.0)] = mask[np.where(mask < 1.0)]
        window_height, window_width = int(compensated_mask.shape[0] * 0.25 / d), int(
            compensated_mask.shape[1] * 0.25 / d)
        windows, windows_anchors = get_windows(compensated_mask, (window_height, window_width), (-1, -1))
        for window_index in range(len(windows)):
            intensity_change = rng.choice([1.0, 0.9, 1.1])  # Make the crack segment equal, darker or lighter
            windows[window_index] = np.minimum(0.90, windows[window_index] * intensity_change)
        compensated_masks.append(join_windows(windows, windows_anchors))
    mask = np.average(np.array(compensated_masks), axis=0)
    mask[np.where(mask == 0.0)] = 1.0

    # Create transitions between crack and background (with a sliding window approach)
    window_height, window_width = int(mask.shape[0] * 0.10), int(mask.shape[1] * 0.10)
    windows, windows_anchors = get_windows(mask, (window_height, window_width), (-1, -1))
    secondary_windows = deepcopy(windows)
    for window_index in range(len(windows)):
        # By using average filters based on circle kernels, we create a fading from background to crack
        se_size = rng.choice([i for i in range(2, 6)])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        windows[window_index] = cv2.filter2D(windows[window_index], -1, kernel / np.sum(kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * se_size, 2 * se_size))
        secondary_windows[window_index] = cv2.filter2D(secondary_windows[window_index], -1, kernel / np.sum(kernel))
    crack_surround_mask = join_windows(windows, windows_anchors)
    secondary_crack_surround_mask = join_windows(secondary_windows, windows_anchors)
    crack_surround_mask = np.minimum(crack_surround_mask, secondary_crack_surround_mask)

    inserted_crack_mask = np.copy(crack_surround_mask)
    ret, gt = cv2.threshold(mask, 0.99991, 1.0, cv2.THRESH_BINARY_INV)  # Get the final crack shape
    # Ensure that the pixels within the final crack mask have the intensity values calculated for the crack
    inserted_crack_mask[np.where(gt == 1.0)] = mask[np.where(gt == 1.0)]

    # Insert the rack into the pavement image
    pavement_image = np.copy(pavement_image)
    for channel in range(3):
        pavement_image[..., channel] *= inserted_crack_mask

    return pavement_image.astype(np.uint8), (255 * gt).astype(np.uint8)
