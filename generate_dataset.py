import argparse
import numpy as np
import os

from distutils.util import strtobool
from image_generation import *
from noisy_labels import compare_gt_stats


def generate_dataset(args):
    # Create rng objects for reproducible outputs
    bg_rng = np.random.default_rng(args.random_seed)  # For background parameters
    crack_rng = np.random.default_rng(args.random_seed)  # For crack parameters
    n_digits = len(str(args.dataset_size))  # For naming purposes
    if not os.path.exists(args.dataset_folder):
        os.makedirs(args.dataset_folder)

    print("Creating images ({:.0f}%)...".format(0), end="\r")
    for i in range(args.dataset_size):

        textures = []  # The background is a mixture of textures
        n_textures = bg_rng.choice([1, 2, 3], p=[0.35, 0.50, 0.15])
        for p in range(n_textures):
            # Create a texture
            pavement = get_pavement(args.image_size,
                                    octaves=bg_rng.integers(3, 6),
                                    scale=bg_rng.normal(args.background_avg_smoothness, args.background_std_smoothness),
                                    persistence=1,
                                    lacunarity=2.0,
                                    base=bg_rng.integers(0, 11))

            # Add color to the pavement
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

            # Add artifacts (crack-like) to the background
            noise_amount_mean = 1 / (10*args.background_avg_smoothness)
            noise_amount = max(0.0, bg_rng.normal(noise_amount_mean, 0.1 * noise_amount_mean))
            rad_std = 2 * max(0.0, 100 * (1.2 * noise_amount_mean - noise_amount))
            noise_it = 4
            for c in range(noise_it):  # Each iteration has different object intensities
                # Add small noise
                pavement = add_cracky_noise(pavement, intensity=bg_rng.normal(args.noise_avg_contrast,
                                                                                 args.noise_std_contrast*1.3),
                                            amount=noise_amount / noise_it, mean_rad=1, std_rad=rad_std,
                                            rng=bg_rng)

            for c in range(noise_it):
                # Add bigger noise
                pavement = add_cracky_noise(pavement, intensity=bg_rng.normal(args.noise_avg_contrast,
                                                                                 args.noise_std_contrast*1.3),
                                            amount=2*noise_amount_mean / noise_it,
                                            mean_rad=2, std_rad=0,
                                            rng=bg_rng)

            # Add salt&pepper noise
            pavement = add_salt_pepper(pavement,
                                       amount=max(0, bg_rng.normal(noise_amount_mean**1.9, noise_amount_mean**2.5)),
                                       s_vs_p=bg_rng.normal(0.7, 0.01),
                                       mean_rad=bg_rng.integers(0, 2),
                                       rng=bg_rng
                                       )

            textures.append(pavement)

        # Join the generated textures to have the final background
        if n_textures > 1:
            pavement = textures[0]
            for p in range(1, n_textures):
                grad = oriented_gradient_fill(pavement.shape[:2],
                                              slope=bg_rng.normal(2.0, 0.2),
                                              center=bg_rng.normal(0.5, 0.05))[..., None]
                grad = np.concatenate((grad, grad, grad), axis=-1)
                pavement = textures[p - 1] * grad + textures[p] * (1.0 - grad)

        crack_maps = []  # The annotation of each added crack will be saved here
        cracked_pavement = np.copy(pavement)
        pavement = pavement.astype(np.float)
        for j in range(args.dataset_size % max(1, round(crack_rng.normal(1, 1))) + 1):  # Random number of cracks
            crack_length = crack_rng.integers(round(args.image_size[0] / 2), args.image_size[0] + 1)
            crack_max_height = crack_rng.integers(round(crack_length / 100), round(crack_length / 20))
            crack_width = max(1, round(crack_rng.normal(args.crack_avg_width, args.crack_std_width)))

            min_y = 0.01 * crack_length  # The minimum y-center value when at the top right position of the crack
            max_rx = (crack_length ** 2) * (1 / (2 * min_y * crack_length - min_y ** 2)) ** 0.5
            # Horizontal radius of an ellipse to create elliptic cracks. Ry is equal to the crack_length.
            # The 1.1 factor is to ensure that the canvas has space to draw the crack shape
            r_x = crack_rng.integers(crack_length * 1.1, int(max_rx) + 1)

            # Get the shape of the crack to add as a mask
            crack_mask = create_crack_shape(crack_length, crack_max_height, r_x,
                                            octaves=crack_rng.integers(8, 10),
                                            scale=crack_rng.normal(10, 0.1),
                                            persistence=crack_rng.normal(0.6, 0.1),
                                            lacunarity=2.0,
                                            base=crack_rng.integers(1, 4),
                                            crack_avg_width=crack_width
                                            )

            # Add a crack to the pavement based on the generated mask
            cracked_pavement, gt = add_crack(cracked_pavement.astype(np.float), crack_mask,
                                             intensity_avg=max(0.0, crack_rng.normal(args.crack_avg_contrast,
                                                                            args.crack_std_contrast)),
                                             orientation=crack_rng.choice([i * 15 for i in range(
                                                 round(360 / 15))]) if crack_width > 1 else crack_rng.choice(
                                                 [i * 90 for i in range(round(360 / 90))])
                                             )
            crack_maps.append(gt)

        # Generate the final annotation for the generated image
        gt = np.zeros(gt.shape, dtype=np.uint8)
        for crack in crack_maps:
            gt = np.maximum(gt, crack)
        # Save both the image and the annotation
        cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}.jpg" % n_digits).format(i + 1)), cracked_pavement)
        cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}_gt.png" % n_digits).format(i + 1)), gt)
        if args.save_background: # Save the background without crack (just if asked by the user)
            cv2.imwrite(os.path.join(args.dataset_folder, ("{:0%sd}_pavement.jpg" % n_digits).format(i + 1)), pavement)
        print("Creating images ({:.0f}%)...".format(100 * (i + 1) / args.dataset_size), end="\r")
    print("Creating images ({:.0f}%)... done!".format(100 * (i + 1) / args.dataset_size))

    # Calculate the number of crack and background pixels. It is saved as a csv file in the dataset folder
    print("Calculating confusion matrix")
    compare_gt_stats(args.dataset_folder, args.dataset_folder, args.dataset_folder)

    # Write (into a text file) the parameters used to generate the current dataset
    output_string = ""
    for attribute in args.__dict__.keys():
        output_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(args.dataset_folder, "generation_parameters.txt"), 'w+') as f:
        f.write(output_string.strip())


def parse_args(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-is", "--image_size", type=int, nargs=2, default=[480, 320],
                        help="Width and height of the database images. e.g. --image_size 420 360")
    parser.add_argument("-f", "--dataset_folder", type=str, default="syncrack_dataset",
                        help="Generated images are saved to this path. If path doesn't exist, it is created.")
    parser.add_argument("-ds", "--dataset_size", type=int, default=200,
                        help="Number of pavement images to generate.")
    parser.add_argument("-cac", "--crack_avg_contrast", type=float, default=0.70,
                        help="A float in the range (0.0, 1.0). The lower the value, the darker the average crack "
                             "intensity with respect to background (higher contrast).")
    parser.add_argument("-csc", "--crack_std_contrast", type=float, default=0.03,
                        help="A float. The standard deviation of the cracks' contrast.")
    parser.add_argument("-nac", "--noise_avg_contrast", type=float, default=0.65,
                        help="A float in the range (0.0, 1.0). The lower the value, the darker the average noise "
                             "intensity with respect to background (higher contrast).")
    parser.add_argument("-nsc", "--noise_std_contrast", type=float, default=0.04,
                        help="A float. The standard deviation of the noise contrast.")
    parser.add_argument("-caw", "--crack_avg_width", type=int, default=2,
                        help="An integer. The approximate average width of the cracks will tend to this number.")
    parser.add_argument("-csw", "--crack_std_width", type=float, default=0.5,
                        help="A float. The width standard deviation between cracks.")
    parser.add_argument("-bas", "--background_avg_smoothness", type=float, default=3.0,
                        help="A float. The higher the value, the more smooth the background will be.")
    parser.add_argument("-bss", "--background_std_smoothness", type=float, default=0.1,
                        help="A float. The standard deviation of the background's smoothness.")
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
