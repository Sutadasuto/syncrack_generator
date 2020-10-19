import argparse
import numpy as np
import os

from image_generation import *


def generate_dataset(image_size, dataset_size, dataset_folder):
    n_digits = len(str(dataset_size))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    print("Creating images ({:.0f}%)...".format(0), end="\r")
    for i in range(dataset_size):

        pavement = get_pavement(image_size,
                                octaves=np.random.randint(3, 8),
                                scale=np.random.normal(3.5, 0.1),
                                persistence=1,
                                lacunarity=2.0,
                                base=np.random.randint(0, 11))

        dark_color_blue = round(np.random.normal(56, 10))
        bright_color_blue = round(np.random.normal(184, 10))
        dark_color = (
            dark_color_blue,
            round(np.random.normal(dark_color_blue - 10, 5)),
            round(np.random.normal(dark_color_blue - 10, 5))
        )
        bright_color = (
            bright_color_blue,
            round(np.random.normal(bright_color_blue - 6, 5)),
            round(np.random.normal(bright_color_blue - 6, 5)))
        pavement = colorize_pavement(pavement, dark_color, bright_color)
        pavement = add_salt_pepper(pavement,
                                   amount=max(0, np.random.normal(0.002, 0.0005)),
                                   s_vs_p=np.random.normal(0.7, 0.01),
                                   mean_rad=np.random.randint(1,3)
                                   )
        crack_maps = []
        for j in range(max(1, round(np.random.normal(1, 1)))):
            crack_length = np.random.randint(round(image_size[0] / 2), image_size[0] + 1)
            crack_max_width = np.random.randint(round(image_size[1] / 10), image_size[1])
            crack_width = 2
            crack_mask = create_crack_shape(crack_length, crack_max_width,
                                            octaves=np.random.randint(6,10),
                                            scale=np.random.normal(4, 0.1),
                                            persistence=0.6,
                                            lacunarity=2.0,
                                            base=np.random.randint(1, 4),
                                            crack_width=crack_width
                                            )
            cracked_pavement, gt = add_crack(pavement, crack_mask,
                                             intensity=np.random.choice([0.55, 0.6, 0.65, 0.7], p=[0.10, 0.15, 0.25, 0.50]),
                                             position=(
                                                 np.random.randint(0, image_size[0] - crack_length + 1),
                                                 np.random.randint(0, image_size[1] - crack_max_width + 1)
                                             ),
                                             orientation=np.random.choice([i*15 for i in range(round(360/15))])
                                         )
            pavement = cracked_pavement.astype(np.float)
            crack_maps.append(gt)
        gt = np.zeros(gt.shape, dtype=np.uint8)
        for crack in crack_maps:
            gt = np.maximum(gt, crack)
        cv2.imwrite(os.path.join(dataset_folder, ("{:0%sd}.jpg" % n_digits).format(i)), cracked_pavement)
        cv2.imwrite(os.path.join(dataset_folder, ("{:0%sd}_gt.png" % n_digits).format(i)), gt)
        # cv2.imwrite(os.path.join(dataset_folder, ("{:0%sd}_pavement.jpg" % n_digits).format(i)), pavement)
        # cv2.imwrite(os.path.join(dataset_folder, ("{:0%sd}_crackshape.jpg" % n_digits).format(i)), crack_mask)
        print("Creating images ({:.0f}%)...".format(100 * (i + 1) / dataset_size), end="\r")
    print("Creating images ({:.0f}%)... done!".format(100 * (i + 1) / dataset_size))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, nargs=2, default=[420, 360],
                        help="Width and height of the database images. e.g. --image_size 420 360")
    parser.add_argument("--dataset_size", type=int, default=500,
                        help="Number of pavement images to generate.")
    parser.add_argument("--dataset_folder", type=str, default="syncrack_dataset",
                        help="Generated images are saved to this path. If path doesn't exist, it is created.")
    args_dict = parser.parse_args(args)
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.image_size, args.dataset_size, args.dataset_folder)