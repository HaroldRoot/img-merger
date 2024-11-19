import argparse
from pathlib import Path

import imagehash
from PIL import Image


def calculate_image_similarity(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    hash1 = imagehash.average_hash(image1)
    hash2 = imagehash.average_hash(image2)

    result = 1 - (hash1 - hash2) / len(hash1.hash) ** 2

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate screenshots vertically.")
    parser.add_argument('--compare', nargs=2,
                        metavar=('image_path1', 'image_path2'),
                        help='Compare the similarity of two images, and the '
                             'paths of the two images need to be provided.')
    return parser.parse_args()


args = parse_args()
if args.compare:
    image_path1 = Path(args.compare[0])
    image_path2 = Path(args.compare[1])
    similarity = calculate_image_similarity(image_path1, image_path2)
    print(f"The similarity of the two images is: {similarity}")
