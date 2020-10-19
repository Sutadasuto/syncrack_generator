import noise
import numpy as np
import cv2

from scipy import ndimage


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


def add_salt_pepper(colorized_pavement, amount=0.002, s_vs_p=0.7, mean_rad=1, std_rad=1):
    out = np.copy(colorized_pavement)
    # Salt mode
    num_salt = int(np.ceil(amount * colorized_pavement.size * s_vs_p))
    salt_color = np.array([colorized_pavement.max() for channel in range(colorized_pavement.shape[-1])])
    coords = [np.random.randint(0, i - 1, num_salt)
              for i in colorized_pavement.shape[:-1]]
    for grain in range(len(coords[0])):
        radius = max(1, round(np.random.normal(mean_rad, std_rad)))
        out = cv2.circle(out, (coords[1][grain], coords[0][grain]), radius, salt_color, thickness=-1)

    # Pepper mode
    num_pepper = int(np.ceil(amount * colorized_pavement.size * (1. - s_vs_p)))
    pepper_color = np.array([colorized_pavement.min() for channel in range(colorized_pavement.shape[-1])])
    coords = [np.random.randint(0, i - 1, num_pepper)
              for i in colorized_pavement.shape[:-1]]
    for grain in range(len(coords[0])):
        radius = max(1, round(np.random.normal(mean_rad, std_rad)))
        out = cv2.circle(out, (coords[1][grain], coords[0][grain]), radius, pepper_color, thickness=-1)
    return 0.7*out + 0.3*colorized_pavement


def create_crack_shape(bounding_length, bounding_width, octaves=8, scale=4, persistence=0.5, lacunarity=2,
                       crack_width=2, base=1):
    x = np.array([round(i * scale) for i in range(int(bounding_length/scale))], dtype=np.int)
    y = np.zeros(x.shape, dtype=np.int)
    canvas = np.zeros((bounding_width, bounding_length), np.uint8)
    for i in range(len(x)):
        y[i] = int(
            # 0.5 * bounding_width * (1 + noise.pnoise1(i / 2**scale, octaves=octaves, persistence=persistence,
            0.5 * bounding_width * (1 + noise.pnoise1(i/len(x), octaves=octaves, persistence=persistence,
                                                      lacunarity=lacunarity, base=base, repeat=bounding_length)))
        if i > 0:
            canvas = cv2.line(canvas, (x[i - 1], y[i - 1]), (x[i], y[i]), color=255, thickness=crack_width)
    return canvas


def add_crack(pavement_image, crack_mask, intensity=0.7, position=(0,0), orientation=0):
    if orientation != 0:
        crack_mask = cv2.morphologyEx(ndimage.rotate(crack_mask, orientation), cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)

    mask = np.ones(pavement_image.shape)[..., 0]
    crack_pixels = np.where(crack_mask == 255)
    for pixel_num in range(len(crack_pixels[0])):
        try:
            mask[position[1] + crack_pixels[0][pixel_num], position[0] + crack_pixels[1][pixel_num]] =\
                np.random.normal(intensity, intensity/10)
        except IndexError:
            continue
    pavement_image = np.copy(pavement_image)
    for channel in range(3):
        pavement_image[..., channel] *= mask

    ret, thresh = cv2.threshold(mask, 0.9*mask.max(), 1.0, cv2.THRESH_BINARY_INV)
    return pavement_image.astype(np.uint8), (255*thresh).astype(np.uint8)


# pavement = get_pavement((420, 360), octaves=5, scale=3.5)
# pavement = colorize_pavement(pavement)
# pavement = add_salt_pepper(pavement)
#
# crack_mask = create_crack_shape(360,60)
# cracked_pavement, gt = add_crack(pavement, crack_mask, position=(0,20), orientation=-30)
#
# cv2.imwrite("pavement.jpg", pavement)
# cv2.imwrite("img.jpg", cracked_pavement)
# cv2.imwrite("gt.png", gt)


