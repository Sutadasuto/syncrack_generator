import argparse
import numpy as np
import os

from distutils.util import strtobool
from image_generation import *
from noisy_labels import compare_gt_stats


def generate_dataset(args):
    # np.random.seed(args.random_seed)
    bg_rng = np.random.default_rng(args.random_seed)
    crack_rng = np.random.default_rng(args.random_seed)
    n_digits = len(str(args.dataset_size))
    if not os.path.exists(args.dataset_folder):
        os.makedirs(args.dataset_folder)

    print("Creating images ({:.0f}%)...".format(0), end="\r")
    for i in range(args.dataset_size):

        textures = []
        n_textures = bg_rng.choice([1, 2, 3], p=[0.35, 0.50, 0.15])
        for p in range(n_textures):
            # pavement = get_pavement(args.image_size,
            #                         octaves=bg_rng.integers(3, 6),
            #                         scale=bg_rng.normal(3.5, 0.1),
            #                         persistence=1,
            #                         lacunarity=2.0,
            #                         base=bg_rng.integers(0, 11))

            pavement = get_pavement(args.image_size,
                                    octaves=bg_rng.integers(3, 6),
                                    scale=bg_rng.normal(args.background_avg_smoothness, args.background_std_smoothness),
                                    persistence=1,
                                    lacunarity=2.0,
                                    base=bg_rng.integers(0, 11))

            dark_color_blue = round(bg_rng.normal(70, 10))
            bright_color_blue = round(bg_rng.normal(180, 10))
            dark_color = (
                dark_color_blue,
                round(bg_rng.normal(dark_color_blue - 10, 3)),
                round(bg_rng.normal(dark_color_blue - 10, 3))
            )
            bright_color = (
                bright_color_blue,
                round(bg_rng.normal(bright_color_blue - 6, 3)),
                round(bg_rng.normal(bright_color_blue - 6, 3)))
            pavement = colorize_pavement(pavement, dark_color, bright_color)

            # noise_amount_mean = 0.025
            noise_amount_mean = 1 / (10*args.background_avg_smoothness)
            noise_amount = max(0.0, bg_rng.normal(noise_amount_mean, 0.1 * noise_amount_mean))
            rad_std = 2 * max(0.0, 100 * (1.2 * noise_amount_mean - noise_amount))
            noise_it = 4
            for c in range(noise_it):
                pavement = add_cracky_noise(pavement, intensity=bg_rng.normal(args.noise_avg_contrast,
                                                                                 args.noise_std_contrast*1.3),
                                            amount=noise_amount / noise_it, mean_rad=1, std_rad=rad_std,
                                            rng=bg_rng)

            for c in range(noise_it):
                # pavement = add_cracky_noise(pavement, intensity=bg_rng.normal(0.75, 0.04), amount=0.05 / noise_it,
                pavement = add_cracky_noise(pavement, intensity=bg_rng.normal(args.noise_avg_contrast,
                                                                                 args.noise_std_contrast*1.3),
                                            amount=2*noise_amount_mean / noise_it,
                                            mean_rad=2, std_rad=0,
                                            rng=bg_rng)

            # pavement = add_salt_pepper(pavement, 0.001, 0.3, 0, 0)
            pavement = add_salt_pepper(pavement,
                                       # amount=max(0, bg_rng.normal(0.001, 0.0001)),
                                       amount=max(0, bg_rng.normal(noise_amount_mean**1.9, noise_amount_mean**2.5)),
                                       s_vs_p=bg_rng.normal(0.7, 0.01),
                                       mean_rad=bg_rng.integers(0, 2),
                                       rng=bg_rng
                                       )

            textures.append(pavement)

        if n_textures > 1:
            pavement = textures[0]
            for p in range(1, n_textures):
                grad = oriented_gradient_fill(pavement.shape[:2],
                                              slope=bg_rng.normal(2.0, 0.2),
                                              center=bg_rng.normal(0.5, 0.05))[..., None]
                grad = np.concatenate((grad, grad, grad), axis=-1)
                pavement = textures[p - 1] * grad + textures[p] * (1.0 - grad)
        crack_maps = []

        cracked_pavement = np.copy(pavement)
        pavement = pavement.astype(np.float)
        for j in range(args.dataset_size % max(1, round(crack_rng.normal(1, 1))) + 1):
            crack_length = crack_rng.integers(round(args.image_size[0] / 2), args.image_size[0] + 1)
            crack_max_width = crack_rng.integers(round(args.image_size[1] / 10), args.image_size[1])
            crack_width = max(1, round(crack_rng.normal(args.crack_avg_width, args.crack_std_width)))
            crack_mask = create_crack_shape(crack_length, crack_max_width,
                                            octaves=crack_rng.integers(8, 10),
                                            scale=crack_rng.normal(10, 0.1),
                                            persistence=crack_rng.normal(0.6, 0.1),
                                            lacunarity=2.0,
                                            base=crack_rng.integers(1, 4),
                                            crack_avg_width=crack_width
                                            )
            cracked_pavement, gt = add_crack(cracked_pavement.astype(np.float), crack_mask,
                                             intensity_avg=max(0.0, crack_rng.normal(args.crack_avg_contrast,
                                                                            args.crack_std_contrast)),
                                             position=(
                                                 crack_rng.integers(0, args.image_size[0] - crack_length + 1),
                                                 crack_rng.integers(0, args.image_size[1] - crack_max_width + 1)
                                             ),
                                             orientation=crack_rng.choice([i * 15 for i in range(
                                                 round(360 / 15))]) if crack_width > 1 else crack_rng.choice(
                                                 [i * 90 for i in range(round(360 / 90))])
                                             )
            crack_maps.append(gt)
        gt = np.zeros(gt.shape, dtype=np.uint8)
        for crack in crack_maps:
            gt = np.maximum(gt, crack)
        cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}.jpg" % n_digits).format(i + 1)), cracked_pavement)
        cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}_gt.png" % n_digits).format(i + 1)), gt)
        if args.save_background:
            cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}_pavement.jpg" % n_digits).format(i + 1)), pavement)
        print("Creating images ({:.0f}%)...".format(100 * (i + 1) / args.dataset_size), end="\r")
    print("Creating images ({:.0f}%)... done!".format(100 * (i + 1) / args.dataset_size))

    print("Calculating confusion matrix")
    compare_gt_stats(args.dataset_folder, args.dataset_folder, args.dataset_folder)
    output_string = ""
    for attribute in args.__dict__.keys():
        output_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(args.dataset_folder, "generation_parameters.txt"), 'w+') as f:
        f.write(output_string.strip())



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-is", "--image_size", type=int, nargs=2, default=[480, 320],
                        help="Width and height of the database images. e.g. --image_size 420 360")
    parser.add_argument("-f", "--dataset_folder", type=str, default="syncrack_dataset",
                        help="Generated images are saved to this path. If path doesn't exist, it is created.")
    parser.add_argument("-ds", "--dataset_size", type=int, default=500,
                        help="Number of pavement images to generate.")
    parser.add_argument("-cac", "--crack_avg_contrast", type=float, default=0.70,
                        help="A float in the range (0.0, 1.0). The lower the value, the darker the average crack "
                             "intensity with respect to background (higher contrast).")
    parser.add_argument("-csc", "--crack_std_contrast", type=float, default=0.03,
                        help="A float. The higher the value, the higher the standard deviation of the cracks' contrast.")
    parser.add_argument("-nac", "--noise_avg_contrast", type=float, default=0.65,
                        help="A float in the range (0.0, 1.0). The lower the value, the darker the average noise "
                             "intensity with respect to background (higher contrast).")
    parser.add_argument("-nsc", "--noise_std_contrast", type=float, default=0.04,
                        help="A float. The higher the value, the higher the standard deviation of the noise contrast.")
    parser.add_argument("-caw", "--crack_avg_width", type=int, default=2,
                        help="An integer. The average approximate width of the cracks will tend to this number.")
    parser.add_argument("-csw", "--crack_std_width", type=float, default=0.5,
                        help="A float. The width standard deviation between cracks.")
    parser.add_argument("-bas", "--background_avg_smoothness", type=float, default=3.0,
                        help="A float. The higher the value, the more smooth the background will be.")
    parser.add_argument("-bss", "--background_std_smoothness", type=float, default=0.1,
                        help="A float. The higher the value, the more smooth the background will be.")
    parser.add_argument("-sb", "--save_background", type=str, default="False",
                        help="If 'True', save the background of each generated image as an additional file.")
    parser.add_argument("-s", "--random_seed", type=int, default=0,
                        help="Seed used for value randomization.")
    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
