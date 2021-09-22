import argparse
import cv2
import numpy as np
import os

from distutils.util import strtobool
from noisy_labels import *


def generate_noisy_labels(args):
    np.random.seed(args.random_seed)
    if args.save_to is None:
        output_folder = args.dataset_folder + "_attacked"
    else:
        output_folder = args.save_to
    comparison_folder = output_folder + "_label_comparison"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)
    string_of_interest = "_gt.png"

    # Get the paths to the ground truth annotations from the dataset to attack
    ground_truth_image_paths = sorted(
        [os.path.join(args.dataset_folder, f) for f in os.listdir(args.dataset_folder)
         if not f.startswith(".") and f.endswith(string_of_interest)],
        key=lambda f: f.lower())

    print("Attacking ground truths ({:.0f}%)...".format(0), end="\r")
    for i, gt_path in enumerate(ground_truth_image_paths):
        new_gt_path = os.path.join(output_folder, os.path.split(gt_path)[1])  # Path to save noisy annotation
        comparison_path = os.path.join(comparison_folder, os.path.split(gt_path)[1])  # Path to save annotation comparison
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # Get noisy annotation
        gt_attacked = attack_annotation(gt, args.operation, args.noise_percentage, args.grid_percentage,
                                        args.dilation_size, args.erosion_size, args.dilation_probability)

        # Copy the input image to the noisy dataset folder
        img_path = gt_path.replace(string_of_interest, ".jpg")
        new_img_path = os.path.join(output_folder, os.path.split(img_path)[1])
        copyfile(img_path, new_img_path)
        cv2.imwrite(new_gt_path, gt_attacked)  # Save the noisy annotation

        # Compare the clean and noisy annotations and save the comparative image
        gt = 255 - gt
        gt_attacked = 255 - gt_attacked
        gt = np.concatenate([gt[..., None] for c in range(3)], axis=-1)
        gt_attacked = np.concatenate([gt_attacked[..., None] for c in range(3)], axis=-1)
        gt_comparison = compare_masks(gt, gt_attacked)
        cv2.imwrite(comparison_path, np.concatenate([gt, gt_attacked, gt_comparison], axis=1))
        print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)), end="\r")
    print("Attacking ground truths ({:.0f}%)...".format(100 * (i + 1) / len(ground_truth_image_paths)))

    # Calculate the confusion matrix of the noisy annotations with respect to the clean ones
    print("Calculating confusion matrix")
    compare_gt_stats(args.dataset_folder, output_folder, comparison_folder)
    output_string = ""
    for attribute in args.__dict__.keys():
        output_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(output_folder, "noise_parameters.txt"), 'w+') as f:
        f.write(output_string.strip())


def parse_args(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset_folder", type=str,
                        help="Folder containing the dataset to attack.")
    parser.add_argument("--save_to", type=str, default=None,
                        help="The noisy dataset will be saved in this folder. It is created if it doesn't exist. If"
                             "'None' is provided, a new folder '$dataset_folder$_attacked' will be created; "
                             "$dataset_folder$ is the path provided as 'dataset_folder' argument.")
    parser.add_argument("-op", "--operation", type=str, default="both",
                        help="Morphological operation used to introduce noise. Either 'dilate', 'erode' or 'both'.")
    parser.add_argument("-np", "--noise_percentage", type=float, default=1.0,
                        help="A float in the range (0.0, 1.0]. This value determines the approximate percentage of the "
                             "annotation image that will be attacked (percentage of windows).")
    parser.add_argument("-gp", "--grid_percentage", type=float, default=0.05,
                        help="A float in the range (0.0, 1.0). To introduce label noise, the annotation image is "
                             "divided into a grid of --grid_percentage*height x --grid_percentage*width windows. The "
                             "morphological operations are performed per window.")
    parser.add_argument("-dp", "--dilation_probability", type=float, default=0.50,
                        help="A float in the range (0.0, 1.0). Ignored if --operation is not 'both'. This value "
                             "determines the probability of choosing dilation as the morphological operation when both"
                             "dilation and erosion are performed.")
    parser.add_argument("-ds", "--dilation_size", type=int, default=3,
                        help="An integer. The structuring element for dilation is a disk with a random diameter using "
                             "this value as average.")
    parser.add_argument("-es", "--erosion_size", type=int, default=2,
                        help="An integer. The structuring element for erosion is a disk with a random diameter using "
                             "this value as average.")
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
    generate_noisy_labels(args)
